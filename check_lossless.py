"""The contract: greedy speculative decoding is BYTE-IDENTICAL to plain decoding.

This is the proof the whole course hangs on (mirrors DeepSpec's own contract
test). If the drafter is dumb we just do more target passes — we NEVER get a
different answer. Run: python check_lossless.py
"""

import torch

from spec_decode import BigramDrafter, plain_greedy, speculative_greedy, tok

PROMPTS = [
    "The key idea behind speculative decoding is",
    "Once upon a time, in a small village",
    "def fibonacci(n):",
]
N = 40

for p in PROMPTS:
    ids = tok(p, return_tensors="pt").input_ids
    base = plain_greedy(ids, N)
    spec, stats = speculative_greedy(ids, N, BigramDrafter(), k=4)
    same = torch.equal(base[:, : ids.shape[1] + N], spec[:, : ids.shape[1] + N])
    rate = stats["accepted"] / max(1, stats["proposed"])
    flag = "OK " if same else "!! MISMATCH"
    print(f"[{flag}] lossless={same}  passes={stats['passes']}/{N}  accept={rate:.0%}  | {p!r}")
    assert same, f"LOSSLESS CONTRACT BROKEN for prompt: {p!r}"

print("\nAll prompts byte-identical to plain decoding. Lossless contract holds.")
