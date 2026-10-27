"""Replace a pack's drafter with a BF16 DFlash2 checkpoint without converting target tensors."""
from __future__ import annotations

import copy
import ctypes
import json
import math
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from model_file import ModelFile, ModelWriter
from model_sections import draft_vocab_bytes


def checkpoint(source: Path, meta: dict) -> tuple[dict, dict]:
    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    dc = cfg.get("dflash_config", {})
    kv = meta.get("gguf_kv", {})
    arch = kv.get("general.architecture", "qwen4exp")
    expected = {"hidden_size": kv.get(arch + ".embedding_length"),
                "num_target_layers": kv.get(arch + ".block_count"),
                "vocab_size": len(kv.get("tokenizer.ggml.tokens", []))}
    if cfg.get("architectures") != ["DFlash2DraftModel"] or cfg.get("dtype") != "bfloat16":
        raise ValueError("DFlash2 source must be a BF16 DFlash2DraftModel")
    for key, value in expected.items():
        if not value or cfg.get(key) != value:
            raise ValueError(f"DFlash2 {key} {cfg.get(key)!r} differs from target {value!r}")
    if (dc.get("block_size"), dc.get("conv_kernel_size"), dc.get("conv_group_size"),
            dc.get("selector_top_k"), dc.get("selector_rank")) != (8, 2, 16, 16, 256):
        raise ValueError("unsupported DFlash2 block, convolution or selector geometry")
    taps = dc.get("target_layer_ids", [])
    if not taps or len(set(taps)) != len(taps) or any(type(t) is not int or not 0 <= t < expected["num_target_layers"] for t in taps):
        raise ValueError("invalid DFlash2 target layer ids")
    if not 0 <= dc.get("mask_token_id", -1) < expected["vocab_size"]:
        raise ValueError("DFlash2 mask token is outside the target vocabulary")
    weights = source / "model.safetensors"
    with weights.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("DFlash2 safetensors header is truncated")
        size, = struct.unpack("<Q", raw)
        if size > 64 << 20:
            raise ValueError("DFlash2 safetensors header is too large")
        header = json.loads(f.read(size))
    intervals = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        shape, offsets = tensor.get("shape", []), tensor.get("data_offsets", [])
        if tensor.get("dtype") != "BF16" or not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f"DFlash2 tensor {name}: expected BF16 with positive dimensions")
        if len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError(f"DFlash2 tensor {name}: invalid offsets")
        lo, hi = offsets
        if lo < 0 or hi - lo != 2 * math.prod(shape) or 8 + size + hi > weights.stat().st_size:
            raise ValueError(f"DFlash2 tensor {name}: invalid byte range")
        intervals.append((lo, hi))
    cursor = 0
    for lo, hi in sorted(intervals):
        if lo != cursor:
            raise ValueError("DFlash2 tensor ranges contain gaps or overlaps")
        cursor = hi
    if not intervals or 8 + size + cursor != weights.stat().st_size:
        raise ValueError("DFlash2 safetensors data size differs from its index")
    n, layers = cfg["hidden_size"], cfg["num_hidden_layers"]
    shapes = {"fc.weight": [n, n * len(taps)], "hidden_norm.weight": [n], "norm.weight": [n],
              "candidate_selector.hidden_projection.weight": [256, n],
              "candidate_selector.predecessor_codebook": [expected["vocab_size"], 256],
              "candidate_selector.successor_codebook": [expected["vocab_size"], 256]}
    for layer in range(layers):
        p = f"layers.{layer}."
        for key in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            shapes[p + key] = [n]
        for key, rows, cols in (("self_attn.q_proj.weight", cfg["num_attention_heads"] * cfg["head_dim"], n),
                                ("self_attn.k_proj.weight", cfg["num_key_value_heads"] * cfg["head_dim"], n),
                                ("self_attn.v_proj.weight", cfg["num_key_value_heads"] * cfg["head_dim"], n),
                                ("self_attn.o_proj.weight", n, cfg["num_attention_heads"] * cfg["head_dim"]),
                                ("mlp.gate_proj.weight", cfg["intermediate_size"], n),
                                ("mlp.up_proj.weight", cfg["intermediate_size"], n),
                                ("mlp.down_proj.weight", n, cfg["intermediate_size"]),
                                ("attention_conv.kernel_projection.weight", n // 16 * 4, n),
                                ("mlp_conv.kernel_projection.weight", n // 16 * 4, n)):
            shapes[p + key] = [rows, cols]
        for key in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            shapes[p + key] = [cfg["head_dim"]]
        for key in ("attention_conv.base_kernel", "mlp_conv.base_kernel"):
            shapes[p + key] = [2, 2, n]
    for name, shape in shapes.items():
        if header.get(name, {}).get("shape") != shape:
            raise ValueError(f"DFlash2 tensor {name}: expected shape {shape}, got {header.get(name, {}).get('shape')}")
    if set(header) - {"__metadata__"} != set(shapes):
        raise ValueError("DFlash2 checkpoint contains unexpected tensors")
    return cfg, header


def quantizer():
    build = Path(__file__).resolve().parents[2] / "build"
    candidates = [build / "libbasalt_quant.so", build / "basalt_quant.dll",
                  build / "Release/basalt_quant.dll", build / "Debug/basalt_quant.dll"]
    library = next((p for p in candidates if p.is_file()), None)
    if library is None:
        raise ValueError("Q4_K conversion requires: cmake --build build --target basalt_quant")
    lib = ctypes.CDLL(str(library))
    fn = lib.basalt_quantize_q4_k
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    fn.restype = ctypes.c_size_t
    return fn


def q4_k(values: np.ndarray, fn) -> bytes:
    x = np.ascontiguousarray(values, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] % 256 or not np.isfinite(x).all():
        raise ValueError("Q4_K requires finite matrix rows divisible by 256")
    rows, cols = x.shape
    result = np.empty(rows * (cols // 256) * 144, dtype=np.uint8)
    written = fn(x.ctypes.data, result.ctypes.data, rows, cols)
    if written != result.nbytes:
        raise ValueError("Q4_K quantizer returned an invalid byte count")
    return result.tobytes()


def quantized_checkpoint(source: Path, header: dict, out: Path) -> None:
    fn = quantizer()
    path = source / "model.safetensors"
    with path.open("rb") as f:
        size, = struct.unpack("<Q", f.read(8))
    base, index, formats, blobs, offset = 8 + size, {}, {}, [], 0
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        lo, hi = tensor["data_offsets"]
        raw = np.memmap(path, mode="r", dtype=np.uint8, offset=base + lo, shape=(hi - lo,))
        shape = tensor["shape"]
        projection = len(shape) == 2 and name.endswith(".weight")
        if projection:
            values = (raw.view("<u2").astype("<u4") << 16).view("<f4").reshape(shape)
            data = q4_k(values, fn)
            stored_shape, dtype = [shape[0], shape[1] // 256 * 144], "U8"
            formats[name] = {"format": "Q4_K", "shape": shape}
        else:
            data, stored_shape, dtype = raw.tobytes(), shape, "BF16"
        index[name] = {"dtype": dtype, "shape": stored_shape, "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
        print(f"drafter {name}: {dtype}, {len(data)} bytes", flush=True)
    index["__metadata__"] = {"basalt.dflash2.quantization": json.dumps(formats, separators=(",", ":"))}
    encoded = json.dumps(index, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with out.open("wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for data in blobs:
            f.write(data)


def proposal_ids(path, cfg: dict) -> list[int] | None:
    """The target-head rows the drafter scores: an int32 list of global token ids, as --draft-vocab writes it."""
    if path is None:
        return None
    vocab = int(cfg.get("vocab_size") or 0)
    ids = np.frombuffer(draft_vocab_bytes(path, vocab or None), dtype="<i4")
    mask = cfg.get("dflash_config", {}).get("mask_token_id")
    if mask is not None and (ids == mask).any():
        raise ValueError("a proposal id is the mask token")
    if np.unique(ids).size != ids.size:
        raise ValueError("a proposal id is repeated")
    return [int(i) for i in ids]


def replace_drafter(base_path: str, source_path: str, out_path: str, quant: str = "bf16",
                    proposal: str | None = None) -> None:
    base, source, out = Path(base_path), Path(source_path), Path(out_path)
    if base.resolve() == out.resolve():
        raise ValueError("DFlash2 output must differ from the source pack")
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    original = ModelFile(str(base))
    meta = copy.deepcopy(json.loads(original.read("meta")))
    cfg, header = checkpoint(source, meta)
    shortlist = proposal_ids(proposal, cfg)
    if quant not in ("bf16", "q4_k"):
        raise ValueError("DFlash2 quantization must be bf16 or q4_k")
    if quant == "q4_k":
        quantizer()
    partial = Path(str(out) + ".partial")
    if partial.exists():
        raise ValueError(f"partial output already exists: {partial}")
    writer = ModelWriter(str(partial))
    try:
        for name in original.names():
            if name == "meta" or name.startswith(("mtp/", "dflash2/")):
                continue
            off, size = original.section(name)
            print(f"copying {name}: {size / (1 << 30):.3f} GiB", flush=True)
            writer.add_file(name, str(base), off, size)
            if writer.entries[-1][3] != original.sections[name][2]:
                raise ValueError(f"source section {name} failed SHA-256 verification")
        if quant == "q4_k":
            with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
                quant_path = Path(tmp) / "model.safetensors"
                quantized_checkpoint(source, header, quant_path)
                writer.add_file("dflash2/model", str(quant_path))
        else:
            writer.add_file("dflash2/model", str(source / "model.safetensors"))
        model_sha = writer.entries[-1][3].hex()
        if shortlist is None:
            writer.add_file("dflash2/config", str(source / "config.json"))
        else:
            # the shortlist rides in the drafter's config: it names head rows, it is not a target weight
            stored = dict(cfg)
            stored["proposal_ids"] = ",".join(map(str, shortlist))
            writer.add_bytes("dflash2/config",
                             json.dumps(stored, separators=(",", ":")).encode())
            print(f"drafter proposal shortlist: {len(shortlist)} head rows of {cfg['vocab_size']}", flush=True)
        if (source / "README.md").is_file():
            writer.add_file("dflash2/model_card", str(source / "README.md"))
        meta["mtp"] = None
        meta["dflash2"] = {"format": "safetensors", "dtype": quant.upper(), "config": cfg,
                            "sha256": model_sha, "license": "apache-2.0"}
        meta["drafter_replacement"] = {"source_meta_sha256": original.sections["meta"][2].hex()}
        writer.add_bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode())
        writer.close()
        os.replace(partial, out)
    except BaseException:
        writer.f.close()
        partial.unlink(missing_ok=True)
        raise
    print(f"done: {quant.upper()} DFlash2, target section hashes unchanged -> {out}", flush=True)
