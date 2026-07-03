"""Chapter 4 — train the DSpark drafter by DISTILLATION, on a laptop CPU.

Recipe: freeze the target. For each position in our tiny corpus, ask the target
"what token would YOU pick next?" (its greedy argmax). Then train the drafter to
predict those same tokens. The drafter learns to imitate the target's greedy path,
so at inference more of its guesses match and get accepted — pure speed, zero risk
to correctness (the verifier still guarantees losslessness).

Heat-safe: we run the big target ONCE per sentence to cache the labels (~30 target
passes total), then train the tiny drafter for a few hundred steps with the target
switched off. Capped at 3 threads so the fan stays calm.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

torch.set_num_threads(3)  # keep the laptop cool

from data import SENTENCES
from dspark_drafter import DSparkDrafter
from spec_decode import BigramDrafter, plain_greedy, speculative_greedy, target, tok

K = 5           # draft block size (DeepSeek ships DSpark-5)
STEPS = 1500
BATCH = 16      # blocks per step — smooths the tiny-corpus gradient noise
LR = 2e-3
SEED = 0

torch.manual_seed(SEED)
g = torch.Generator().manual_seed(SEED)

# --- Cache the target's greedy next-token labels once (the only heavy step). ---
print("caching target labels (one pass per sentence)...")
cache = []  # (ids (1,T), tgt_argmax (1,T))
with torch.no_grad():
    for s in SENTENCES:
        ids = tok(s, return_tensors="pt").input_ids
        if ids.shape[1] < K + 2:
            continue
        tgt_argmax = target(ids).logits.argmax(-1)  # (1,T): position p -> token at p+1
        cache.append((ids, tgt_argmax))
print(f"cached {len(cache)} sentences")


def acceptance(drafter, prompts=("The key idea behind speculative decoding is",
                                 "The capital of France is")):
    """Quick acceptance% + passes on a couple of held-out prompts (greedy)."""
    tot_acc = tot_prop = tot_pass = tot_new = 0
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        _, st = speculative_greedy(ids, 32, drafter, k=K)
        tot_acc += st["accepted"]; tot_prop += st["proposed"]
        tot_pass += st["passes"]; tot_new += 32
    return tot_acc / max(1, tot_prop), tot_pass, tot_new


# --- Train the drafter. --------------------------------------------------------
drafter = DSparkDrafter(target, block=K).train()
opt = torch.optim.Adam(drafter.parameters(), lr=LR)

before = acceptance(drafter)
print(f"\nBEFORE training  accept={before[0]:.0%}  passes={before[1]}/{before[2]}")

def sample_batch(b):
    """Draw b random K-blocks from the cached corpus (mixed sentences)."""
    cw, pv, lb = [], [], []
    for _ in range(b):
        ids, tgt_argmax = cache[int(torch.randint(len(cache), (1,), generator=g))]
        T = ids.shape[1]
        s = int(torch.randint(0, T - K, (1,), generator=g))
        cw.append(drafter.window_of(ids[:, :s + 1])[0])  # (W,) context ending at s
        pv.append(ids[0, s:s + K])
        lb.append(tgt_argmax[0, s:s + K])
    return torch.stack(cw), torch.stack(pv), torch.stack(lb)  # (b,W),(b,K),(b,K)


for step in range(1, STEPS + 1):
    ctx_win, prev_block, labels = sample_batch(BATCH)
    logits = drafter.block_logits_teacher_forced(ctx_win, prev_block)  # (b,K,V)
    loss = F.cross_entropy(logits.reshape(-1, drafter.vocab), labels.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0:
        print(f"  step {step:4d}/{STEPS}  loss {loss.item():.3f}")

drafter.eval()
after = acceptance(drafter)
print(f"\nAFTER  training  accept={after[0]:.0%}  passes={after[1]}/{after[2]}")

# Bigram baseline for contrast (chapter 2's dumb drafter).
big = acceptance(BigramDrafter())
print(f"(bigram baseline accept={big[0]:.0%}  passes={big[1]}/{big[2]})")

torch.save(drafter.state_dict(), "drafter.pt")
print("\nsaved trained drafter -> drafter.pt")
