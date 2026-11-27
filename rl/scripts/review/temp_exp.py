"""Temperature what-if: 4 seats of the net acting at temperature TAU (0 = greedy), 4 seats sampling as trained.

    python temp_exp.py TAU SEED0 N
"""

import os
import random
import statistics as st
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)
from seaglass_rl import checkpoint, play
from seaglass_rl.ladder import checkpoint_options
from seaglass_rl.play import GameSpec, LobbyConfig

NET = os.environ.get("NET", "runs/capacity/small/latest.pt")


class Tempered:
    def __init__(self, model, tau, seed):
        self.model = model
        self.tau = tau
        self.rng = np.random.default_rng(seed)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def act_batch(self, pv, ci, nf, mk, deterministic=False):
        with torch.no_grad():
            logits, values = self.model.forward(
                torch.from_numpy(np.ascontiguousarray(pv, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(ci, dtype=np.int64)),
                torch.from_numpy(np.ascontiguousarray(nf, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(mk, dtype=np.bool_)),
            )
            lps = torch.log_softmax(logits, dim=-1).numpy()
        B = lps.shape[0]
        if self.tau <= 0:
            acts = lps.argmax(axis=-1)
        else:
            z = lps / self.tau
            z = z - z.max(axis=-1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=-1, keepdims=True)
            acts = np.array([self.rng.choice(p.shape[1], p=p[i]) for i in range(B)])
        return acts, lps[np.arange(B), acts], values.numpy()


def main():
    tau, seed0, n = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    model, _ = checkpoint.load(NET)
    model._np_rng = np.random.default_rng(seed0)
    opts = checkpoint_options(NET)
    models = {"cool": Tempered(model, tau, seed0 + 1), "rival": model}
    cfg = LobbyConfig()
    specs = []
    for i in range(n):
        ctrl = ["cool"] * 4 + ["rival"] * 4
        random.Random(seed0 + i).shuffle(ctrl)
        specs.append(GameSpec(seed=seed0 + i, controllers=ctrl))
    lob = []
    for k in range(0, n, 16):
        for res in play.play_games(cfg, specs[k : k + 16], models, opts, record=()):
            lob.append(st.mean(res.placements[u] for u, c in enumerate(res.controllers) if c == "cool"))
    print(f"tau {tau}: avg place {st.mean(lob):.3f} +- {st.stdev(lob) / len(lob) ** 0.5:.3f} over {len(lob)} lobbies (4.5 level; rival sampled at 1.0)", flush=True)


if __name__ == "__main__":
    main()
