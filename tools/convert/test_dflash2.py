"""Drafter replacement preserves target bytes and validates the checkpoint before writing."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_file import ModelFile, ModelWriter
from dflash2 import replace_drafter, quantizer, q4_k
import numpy as np
import ggufq


class Dflash2Pack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "draft"
        self.source.mkdir()
        self.cfg = {"architectures": ["DFlash2DraftModel"], "dtype": "bfloat16", "hidden_size": 256,
                    "num_target_layers": 48, "vocab_size": 64, "num_hidden_layers": 1,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
                    "intermediate_size": 768, "sliding_window": 2048,
                    "dflash_config": {"block_size": 8, "conv_kernel_size": 2, "conv_group_size": 16,
                                      "selector_top_k": 16, "selector_rank": 256,
                                      "target_layer_ids": [5, 15, 25, 35, 45], "mask_token_id": 7}}
        self.write_checkpoint()
        self.base, self.out = self.root / "base.basalt", self.root / "new.basalt"
        writer = ModelWriter(str(self.base))
        for name, data in (("gguf", b"target tensor bytes"), ("experts", b"expert bytes"),
                           ("mtp/experts", b"old drafter"), ("vision/mmproj", b"vision bytes")):
            writer.add_bytes(name, data)
        writer.add_bytes("meta", json.dumps({"gguf_kv": {"general.architecture": "qwen4exp",
                              "qwen4exp.embedding_length": 256, "qwen4exp.block_count": 48,
                              "tokenizer.ggml.tokens": [str(i) for i in range(64)]},
                              "tensors": {"test": {"offset": 123}}, "mtp": {"expert_format": "q2_0"}}).encode())
        writer.close()

    def write_checkpoint(self):
        (self.source / "config.json").write_text(json.dumps(self.cfg))
        shapes = {"fc.weight": [256, 1280], "hidden_norm.weight": [256], "norm.weight": [256],
                  "candidate_selector.hidden_projection.weight": [256, 256],
                  "candidate_selector.predecessor_codebook": [64, 256],
                  "candidate_selector.successor_codebook": [64, 256]}
        for name in ("input_layernorm", "post_attention_layernorm"):
            shapes[f"layers.0.{name}.weight"] = [256]
        for name in ("q", "k"):
            shapes[f"layers.0.self_attn.{name}_norm.weight"] = [64]
        for name, shape in {"self_attn.q_proj.weight": [256, 256], "self_attn.k_proj.weight": [128, 256],
                            "self_attn.v_proj.weight": [128, 256], "self_attn.o_proj.weight": [256, 256],
                            "mlp.gate_proj.weight": [768, 256], "mlp.up_proj.weight": [768, 256],
                            "mlp.down_proj.weight": [256, 768]}.items():
            shapes["layers.0." + name] = shape
        for name in ("attention_conv", "mlp_conv"):
            shapes[f"layers.0.{name}.kernel_projection.weight"] = [64, 256]
            shapes[f"layers.0.{name}.base_kernel"] = [2, 2, 256]
        header, data = {}, bytearray()
        for name, shape in shapes.items():
            size = 2
            for n in shape:
                size *= n
            header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [len(data), len(data) + size]}
            data.extend(b"\x80\x3f" * (size // 2))
        raw = json.dumps(header).encode()
        (self.source / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + data)

    def test_target_sections_and_bf16_checkpoint_are_identical(self):
        before = self.base.read_bytes()
        replace_drafter(str(self.base), str(self.source), str(self.out))
        old, new = ModelFile(str(self.base)), ModelFile(str(self.out))
        for name in ("gguf", "experts", "vision/mmproj"):
            self.assertEqual(old.read(name), new.read(name))
            self.assertEqual(old.sections[name][2], new.sections[name][2])
        self.assertFalse(any(n.startswith("mtp/") for n in new.names()))
        self.assertEqual(new.read("dflash2/model"), (self.source / "model.safetensors").read_bytes())
        self.assertTrue(all(new.verify(n) for n in new.names()))
        self.assertIsNone(json.loads(new.read("meta"))["mtp"])
        self.assertEqual(json.loads(new.read("meta"))["tensors"], {"test": {"offset": 123}})
        self.assertEqual(self.base.read_bytes(), before)

    def test_incompatible_config_leaves_no_output(self):
        self.cfg["vocab_size"] = 65
        self.write_checkpoint()
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            replace_drafter(str(self.base), str(self.source), str(self.out))
        self.assertFalse(self.out.exists())
        self.assertFalse(Path(str(self.out) + ".partial").exists())

    def test_q4_k_pack_keeps_codebooks_and_target_bytes(self):
        try:
            quantizer()
        except ValueError as e:
            self.skipTest(str(e))
        replace_drafter(str(self.base), str(self.source), str(self.out), "q4_k")
        old, new = ModelFile(str(self.base)), ModelFile(str(self.out))
        for name in ("gguf", "experts", "vision/mmproj"):
            self.assertEqual(old.read(name), new.read(name))
        raw = new.read("dflash2/model")
        size, = struct.unpack_from("<Q", raw)
        header = json.loads(raw[8:8 + size])
        formats = json.loads(header["__metadata__"]["basalt.dflash2.quantization"])
        self.assertEqual(formats["fc.weight"], {"format": "Q4_K", "shape": [256, 1280]})
        self.assertEqual(header["fc.weight"]["dtype"], "U8")
        self.assertEqual(header["fc.weight"]["shape"], [256, 720])
        for name in ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook", "hidden_norm.weight"):
            self.assertEqual(header[name]["dtype"], "BF16")
            lo, hi = header[name]["data_offsets"]
            self.assertEqual(raw[8 + size + lo:8 + size + hi], b"\x80\x3f" * ((hi - lo) // 2))
        self.assertEqual(json.loads(new.read("meta"))["dflash2"]["dtype"], "Q4_K")
        self.assertTrue(all(new.verify(n) for n in new.names()))

    def test_q4_k_quantization_error_and_invalid_inputs(self):
        try:
            fn = quantizer()
        except ValueError as e:
            self.skipTest(str(e))
        rng = np.random.default_rng(95)
        x = rng.standard_normal((8, 256)).astype(np.float32)
        raw = q4_k(x, fn)
        back = ggufq.dequantize(np.frombuffer(raw, np.uint8).reshape(8, 144), ggufq.GGMLQuantizationType.Q4_K)
        rms = np.sqrt(np.mean((back - x) ** 2) / np.mean(x ** 2))
        self.assertLess(rms, 0.10)
        zeros = np.frombuffer(q4_k(np.zeros_like(x), fn), np.uint8).reshape(8, 144)
        self.assertTrue(np.all(ggufq.dequantize(zeros, ggufq.GGMLQuantizationType.Q4_K) == 0))
        for bad in (x[:, :255], np.full_like(x, np.nan)):
            with self.assertRaises(ValueError):
                q4_k(bad, fn)

    def test_truncated_weights_leave_no_output(self):
        path = self.source / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "byte range"):
            replace_drafter(str(self.base), str(self.source), str(self.out))
        self.assertFalse(self.out.exists())

    def test_corrupt_target_section_is_not_copied(self):
        original = ModelFile(str(self.base))
        with self.base.open("r+b") as f:
            f.seek(original.section("experts")[0])
            f.write(b"X")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            replace_drafter(str(self.base), str(self.source), str(self.out))
        self.assertFalse(self.out.exists())
        self.assertFalse(Path(str(self.out) + ".partial").exists())

    def test_source_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, "differ"):
            replace_drafter(str(self.base), str(self.source), str(self.base))


if __name__ == "__main__":
    unittest.main()
