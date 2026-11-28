"""DeepSeek4 in the engine's --serve loop: the two moves a server makes between requests of one conversation.

- A reply that ends on its end token inside a decode window leaves the session a row past what the engine keeps; the next
  read of the conversation starts there and goes back over that row.
- A read stopped as it starts (STOP after RESUME) ends as cancelled, and the engine answers the next requests: a short
  prompt, and the stopped prompt in full. (Greedy tokens are not compared with a fresh engine's: a long read moves the
  expert tier, and the experts the CPU computes round differently from the GPU's.)

Extra arguments after the pack go to the engine (--gpu 0,1 --gpu-mode peer runs it on two cards)."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

LONG = 3992    # a prompt read in one chunk, as long as the one a server once stopped


def ids(path: Path) -> list[int]:
    return [int(x) for x in path.read_text().replace(",", " ").split()]


class Engine:
    def __init__(self, cmd: list[str]):
        self.log = tempfile.TemporaryFile("w+")     # the start's log is long: a pipe would fill and hold the engine
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                  bufsize=1)
        for line in self.p.stdout:
            if line.startswith("READY"):
                return
        raise AssertionError("the engine did not start: " + self.tail())

    def tail(self) -> str:
        self.log.seek(0)
        return self.log.read()[-3000:]

    def gen(self, n: int, prompt: list[int], stop_at_start: bool = False) -> tuple[list[int], str]:
        self.p.stdin.write("GEN %d temperature=0 %s\n" % (n, ",".join(map(str, prompt))))
        self.p.stdin.flush()
        got = []
        for line in self.p.stdout:
            if line.startswith("RESUME ") and stop_at_start:
                self.p.stdin.write("STOP\n")   # the read starts after RESUME: the stop is seen before its first chunk
                self.p.stdin.flush()
            elif line.startswith("T "):
                got.append(int(line.split()[1]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                return got, line.strip()
        raise AssertionError("the engine ended: " + self.tail())

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()
        self.log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixtures", required=True, help="data/bench/deepseek4")
    args, extra = parser.parse_known_args()
    if not Path(args.model).is_file():
        print("SKIP: model pack unavailable")
        return 77
    fx = Path(args.fixtures)
    short, long_ = ids(fx / "struct.ids"), ids(fx / "prefill.ids")[:LONG]
    e = Engine([str(Path(args.engine).resolve()), "--serve", "--model", str(Path(args.model).resolve()),
                "--max-context", "16384", *extra])
    try:
        out, done = e.gen(200, short)
        assert done.startswith("DONE") and " stop " in done, f"the short prompt should end on its end token: {done}"
        _, done = e.gen(16, short + out + [271, 13])
        assert done.startswith("DONE"), f"the read after a reply that stopped inside a window: {done}"
        _, done = e.gen(16, long_, stop_at_start=True)
        assert done.startswith("DONE") and " cancel " in done, f"a read stopped as it starts: {done}"
        again, done = e.gen(32, short)
        assert done.startswith("DONE") and again, f"the request after a stopped read: {done}"
        full, done = e.gen(8, long_)
        assert done.startswith("DONE") and full, f"the stopped prompt in full: {done}"
    finally:
        e.close()
    print("serve steps: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
