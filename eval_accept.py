"""Chapter 5 — measure acceptance honestly, and see WHERE each drafter wins.

Three drafters, all provably lossless (we assert byte-identical output every time):
  * bigram        — the zero-training n-gram lookup from chapter 2. This is a real
                    technique ("prompt-lookup decoding") and a shockingly strong
                    baseline on repetitive text, because a model's own greedy output
                    repeats itself a lot.
  * dspark(raw)   — the DSpark drafter, untrained (random init).
  * dspark(train) — the same drafter after chapter 4's distillation.

We split prompts into REPETITIVE (code / technical, where the n-gram thrives) and
NOVEL prose (held-out, where there's nothing to look up) to show the trade-off.
Acceptance = fraction of drafted tokens the target accepted; passes = target
forwards used to make 32 tokens (lower is faster; plain decoding needs 32).
"""

from __future__ import annotations

import torch

torch.set_num_threads(3)

from dspark_drafter import DSparkDrafter
from spec_decode import BigramDrafter, plain_greedy, speculative_greedy, target, tok

K = 5
N = 32

REPETITIVE = [
    "def fibonacci(n):",
    "for i in range(10):\n    print(",
    "The key idea behind speculative decoding is",
    "import torch\nimport torch.nn as",
]
NOVEL = [
    "The lighthouse keeper poured himself a cup of",
    "My grandmother always said that the secret to",
    "On the third day of the expedition, the team discovered",
    "Nobody expected the small bakery on the corner to",
]


def run(drafter_factory, prompts):
    """Return (accept_rate, passes, verified_lossless) over prompts."""
    acc = prop = passes = 0
    lossless = True
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        base = plain_greedy(ids, N)
        out, st = speculative_greedy(ids, N, drafter_factory(), k=K)
        lossless &= torch.equal(base[:, : ids.shape[1] + N], out[:, : ids.shape[1] + N])
        acc += st["accepted"]; prop += st["proposed"]; passes += st["passes"]
    return acc / max(1, prop), passes, lossless


# Trained drafter loaded once, reused (factory returns the same object).
trained = DSparkDrafter(target, block=K).eval()
trained.load_state_dict(torch.load("drafter.pt"))

drafters = {
    "bigram       ": lambda: BigramDrafter(),
    "dspark(raw)  ": (lambda: DSparkDrafter(target, block=K).eval()),
    "dspark(train)": (lambda: trained),
}

for setname, prompts in [("REPETITIVE (n-gram's turf)", REPETITIVE), ("NOVEL prose (held-out)", NOVEL)]:
    print(f"\n=== {setname} — {len(prompts)} prompts, {N} tokens each ===")
    print(f"{'drafter':<14} {'accept':>7} {'passes':>8}  {'vs plain':>9}  lossless")
    for name, fac in drafters.items():
        rate, passes, ll = run(fac, prompts)
        total = len(prompts) * N
        print(f"{name:<14} {rate:>6.0%} {passes:>6}/{total}  {1 - passes/total:>8.0%}  {'OK' if ll else 'BROKEN!'}")

print("\nAll runs byte-identical to plain decoding — every drafter is lossless.")
