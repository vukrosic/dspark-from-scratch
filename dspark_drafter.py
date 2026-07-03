"""DSpark's real drafter, from scratch (chapter 3).

The bigram drafter was dumb: it only knew "what token usually follows this one".
DSpark's drafter is the clever, cheap guesser DeepSeek actually ships. Two ideas:

  1. SEMI-AUTOREGRESSIVE. To propose a block of k tokens, a normal draft model
     would run k serial forward passes (slow). DSpark instead runs ONE cheap
     "backbone" pass that produces a base guess for all k positions at once
     (parallel -> fast), then threads a tiny low-rank "Markov" head through them
     that only looks at the immediately preceding token (sequential -> accurate
     deep into the block). Parallel gives speed + a strong first token; the
     sequential thread stops acceptance decaying further into the block.

  2. IT REUSES THE FROZEN TARGET. The drafter borrows the target's own token
     embedding and output head (they're tied) and adds only a handful of small
     matrices on top. So it's tiny to store and tiny to run — a couple of matmuls
     versus a full 30-layer target forward.

Nothing about the VERIFIER changes — we still use the exact lossless accept rule
from chapters 1-2. A better drafter only means more guesses get accepted, i.e.
more speed. It can never change the answer. (That's why we can distill it however
we like in chapter 4 without ever risking correctness.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSparkDrafter(nn.Module):
    def __init__(self, target, block: int = 5, markov_rank: int = 64, window: int = 8):
        super().__init__()
        dim = int(target.config.hidden_size)
        self.dim = dim
        self.vocab = int(target.config.vocab_size)
        self.block = max(1, int(block))
        self.window = max(1, int(window))  # backbone sees last `window` tokens (cheap mean, no attention)
        r = max(1, int(markov_rank))

        # Parallel backbone: one cheap pass -> a base hidden for ALL k positions.
        # base_ctx summarises the last context token; pos_emb shifts it per future
        # position; base_mlp mixes them. No recurrence over the block => parallel.
        self.base_ctx = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.pos_emb = nn.Parameter(torch.zeros(self.block, dim))
        self.base_mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

        # Low-rank Markov head: a rank-r nudge driven ONLY by the preceding token,
        # added to the parallel base. This is the cheap sequential thread.
        self.markov_down = nn.Linear(dim, r, bias=False)
        self.markov_up = nn.Linear(r, dim, bias=False)

        self.head_norm = nn.LayerNorm(dim)   # match the scale the output head wants
        self.conf_head = nn.Linear(dim, 1)   # per-position P(accept), used later

        # Borrowed FROZEN target parts — kept as plain refs so they are NOT
        # registered as drafter parameters and never trained.
        object.__setattr__(self, "_emb", target.get_input_embeddings())
        object.__setattr__(self, "_lm_head", target.lm_head)

    def _window_ctx(self, ctx_win: torch.Tensor) -> torch.Tensor:
        """ctx_win (B,W) token ids -> context vector (B,dim). Cheap mean of the last
        W token embeddings — gives the backbone real context without any attention."""
        return self.base_ctx(self._emb(ctx_win).mean(dim=1))

    def window_of(self, ids: torch.Tensor) -> torch.Tensor:
        """Last `window` tokens of ids (B,T) -> (B,W), left-padded if too short."""
        W = self.window
        win = ids[:, -W:]
        if win.shape[1] < W:
            pad = win[:, :1].expand(-1, W - win.shape[1])
            win = torch.cat([pad, win], dim=1)
        return win

    # --- shared math: base + markov -> logits for a block ---------------------
    def _heads(self, ctx_win: torch.Tensor, prev_toks: torch.Tensor):
        """ctx_win (B,W) context tokens, prev_toks (B,k) -> logits (B,k,vocab), h.

        `prev_toks[:, j]` is the token immediately before block position j.
        In TRAINING these are the true tokens (parallel, teacher-forced).
        In INFERENCE they are filled one at a time as we draft (see `propose`).
        """
        k = prev_toks.shape[1]
        ctx = self._window_ctx(ctx_win)                                   # (B,dim)
        bases = self.base_mlp(ctx[:, None, :] + self.pos_emb[None, :k, :]) # (B,k,dim)
        adj = self.markov_up(F.gelu(self.markov_down(self._emb(prev_toks))))  # (B,k,dim)
        h = self.head_norm(bases + adj)                                   # (B,k,dim)
        return self._lm_head(h), h

    def block_logits_teacher_forced(self, ctx_win, prev_toks):
        """Parallel forward used in chapter 4 training (prev = true tokens)."""
        return self._heads(ctx_win, prev_toks)[0]

    @torch.no_grad()
    def propose(self, ids: torch.Tensor, k: int) -> torch.Tensor:
        """Greedy: propose k tokens (B,k). ONE backbone pass + k tiny Markov steps."""
        emb = self._emb
        ctx = self._window_ctx(self.window_of(ids))                       # (B,dim)
        bases = self.base_mlp(ctx[:, None, :] + self.pos_emb[None, :k, :])
        out, prev = [], ids[:, -1]
        for j in range(k):
            adj = self.markov_up(F.gelu(self.markov_down(emb(prev))))
            h = self.head_norm(bases[:, j, :] + adj)
            nid = self._lm_head(h).argmax(-1)                            # greedy pick
            out.append(nid)
            prev = nid  # Markov: next position sees only this fresh token
        return torch.stack(out, dim=1)                                   # (B,k)
