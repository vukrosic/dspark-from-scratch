"""Speculative decoding from scratch — the lossless core (chapters 1-2).

The whole idea: generating text one token at a time is slow because each token
needs a full forward pass of a big model. Speculative decoding has a CHEAP
"drafter" guess the next few tokens, then the big "target" model checks all the
guesses in ONE forward pass, keeping only the prefix it agrees with. You get
several tokens per target pass instead of one — and, done right, the output is
*byte-identical* to plain decoding. That guarantee is the point of this file.

Torch-only, CPU, no framework. Target = a real small HF model, frozen. The
drafter here is deliberately dumb (a bigram table built from the text so far) —
losslessness holds no matter how bad the drafter is; a better drafter only makes
it FASTER, never more correct. (That's exactly what DSpark improves later:
a smarter, cheaper drafter. The verifier below is unchanged from vanilla
speculative sampling, which DSpark reuses.)
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M"  # already cached locally; runs on CPU

print(f"loading target: {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL)
target = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()


@torch.no_grad()
def target_logits(ids: torch.Tensor) -> torch.Tensor:
    """One target forward. ids (1, T) -> logits (1, T, vocab)."""
    return target(ids).logits


# --- Baseline: plain greedy decoding, one token per target forward. -----------
@torch.no_grad()
def plain_greedy(ids: torch.Tensor, n: int) -> torch.Tensor:
    for _ in range(n):
        nxt = target_logits(ids)[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


# --- The dumbest drafter that works: a bigram table over the text so far. ------
class BigramDrafter:
    """Guess the next token = whatever most recently followed the current token.

    No model, no training — just a dict updated as text streams by. Terrible
    accuracy, which is the point: it stresses the verifier and still stays
    lossless. Chapter 3 replaces this with DSpark's real semi-autoregressive head.
    """

    def __init__(self) -> None:
        self.nxt: dict[int, int] = {}

    def observe(self, ids: torch.Tensor) -> None:
        row = ids[0].tolist()
        for a, b in zip(row, row[1:]):
            self.nxt[a] = b  # last-seen successor wins

    def propose(self, ids: torch.Tensor, k: int) -> torch.Tensor:
        self.observe(ids)
        cur = int(ids[0, -1])
        out = []
        for _ in range(k):
            g = self.nxt.get(cur, cur)  # fallback: repeat current token
            out.append(g)
            cur = g
        return torch.tensor([out], dtype=torch.long)


# --- The lossless verifier (greedy / temperature 0). --------------------------
@torch.no_grad()
def speculative_greedy(ids: torch.Tensor, n: int, drafter: BigramDrafter, k: int = 4):
    """Generate n tokens, lossless-identical to plain_greedy, but in fewer passes.

    Per round: drafter proposes k tokens; the target scores the whole block in ONE
    forward; we accept the longest prefix where the draft matches the target's own
    argmax, then commit ONE correction token (the target's argmax at the first
    mismatch — exactly what plain greedy would have emitted there). So every round
    commits (accepted + 1) tokens for a single target forward.
    """
    produced = 0
    proposed_total = 0
    accepted_total = 0
    passes = 0
    while produced < n:
        draft = drafter.propose(ids, k)                     # (1, k)
        cand = torch.cat([ids, draft], dim=1)
        logits = target_logits(cand)                        # (1, T+k, vocab)
        passes += 1
        t_arg = logits[:, -(k + 1):].argmax(-1)             # (1, k+1) target's picks

        acc = 0
        for j in range(k):
            proposed_total += 1
            if int(draft[0, j]) == int(t_arg[0, j]):
                acc += 1
            else:
                break
        accepted_total += acc
        # accepted prefix + the target's correction token at position `acc`
        commit = torch.cat([draft[:, :acc], t_arg[:, acc:acc + 1]], dim=1)
        ids = torch.cat([ids, commit], dim=1)
        produced += acc + 1
    return ids, dict(proposed=proposed_total, accepted=accepted_total, passes=passes)


if __name__ == "__main__":
    prompt = "The key idea behind speculative decoding is"
    ids = tok(prompt, return_tensors="pt").input_ids
    N = 40

    base = plain_greedy(ids, N)
    spec, stats = speculative_greedy(ids, N, BigramDrafter(), k=4)

    print("\nprompt:", prompt)
    print("target :", tok.decode(base[0, ids.shape[1]:]))
    acc_rate = stats["accepted"] / max(1, stats["proposed"])
    print(f"\nspec passes: {stats['passes']} (plain would need {N})")
    print(f"acceptance : {stats['accepted']}/{stats['proposed']} = {acc_rate:.0%}"
          "  (dumb bigram drafter — DSpark's real drafter goes here in ch3)")
