"""Forced-levelling what-if: 4 seats of the net made to level on a human curve, 4 seats of the plain net.

    python curve_exp.py CURVE SEED0 N OUT_DIR

The "curve" seats are the same network, except that whenever the Tavern upgrade is legal and the seat's
Tavern Tier is below the curve's target for the turn, the upgrade is taken. Everything else is the net's.
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import watch as w  # patches play._apply for logging
from seaglass_rl import checkpoint, play
from seaglass_rl.actions import ACTION_TO_INDEX
from seaglass_rl.ladder import checkpoint_options
from seaglass_rl.play import GameSpec, LobbyConfig

UPGRADE = ACTION_TO_INDEX[("upgrade_tavern", None, None)]

# turn -> target Tavern Tier
CURVES = {
    "standard": {2: 2, 5: 3, 7: 4, 9: 5, 11: 6},
    "fast": {2: 2, 4: 3, 6: 4, 8: 5, 10: 6},
    "slow": {2: 2, 5: 3, 8: 4, 10: 5, 12: 6},
    "to4": {2: 2, 5: 3, 7: 4},
    "to5": {2: 2, 5: 3, 7: 4, 9: 5},
    "late5": {9: 4, 11: 5, 13: 6},
}


class Forced:
    """Wraps a model: same forward, with the upgrade forced on the curve."""

    def __init__(self, model, curve):
        self.model = model
        self.curve = sorted(curve.items())

    def __getattr__(self, name):
        return getattr(self.model, name)

    def target(self, turn):
        t = 1
        for at, tier in self.curve:
            if turn >= at:
                t = tier
        return t

    def act_batch(self, pv, ci, nf, mk, deterministic=False):
        acts, lps, vals = self.model.act_batch(pv, ci, nf, mk, deterministic=deterministic)
        acts = np.array(acts)
        for i in range(len(acts)):
            if mk[i, UPGRADE]:
                turn = int(round(float(pv[i, 0]) * 20))
                tier = int(round(float(pv[i, 3]) * 6))
                if tier < self.target(turn):
                    acts[i] = UPGRADE
        return acts, lps, vals


def main():
    curve, seed0, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(out, exist_ok=True)
    model, _ = checkpoint.load(w.NET)
    model._np_rng = np.random.default_rng(seed0)
    opts = checkpoint_options(w.NET)
    opts.keep_replays = True
    models = {"curve": Forced(model, CURVES[curve]), "rival": model}
    cfg = LobbyConfig()
    places = []
    for i in range(n):
        seed = seed0 + i
        path = os.path.join(out, f"curve-{curve}_{seed}.json")
        if os.path.exists(path):
            continue
        ctrl = ["curve"] * 4 + ["rival"] * 4
        random.Random(seed).shuffle(ctrl)
        w.GAMES.clear()
        w.LOG.clear()
        t0 = time.time()
        res = play.play_games(cfg, [GameSpec(seed=seed, controllers=ctrl)], models, opts, record=(), on_turn_end=w.on_turn_end)[0]
        (rec,) = w.GAMES.values()
        rec.update({"seed": seed, "mode": f"curve-{curve}", "search": "none", "controllers": ctrl, "placements": res.placements, "turns": res.turns})
        rec["players"] = {str(k): v for k, v in rec["players"].items()}
        json.dump(rec, open(path, "w"))
        res.save_replay(os.path.join(out, f"curve-{curve}_{seed}.replay.pb"))
        places += [res.placements[u] for u, c in enumerate(ctrl) if c == "curve"]
    if places:
        m = sum(places) / len(places)
        # clustered by lobby: 4 seats per lobby
        lob = [sum(places[i : i + 4]) / 4 for i in range(0, len(places), 4)]
        import statistics as st

        se = st.stdev(lob) / len(lob) ** 0.5 if len(lob) > 1 else 0
        print(f"curve {curve}: forced seats avg place {m:.3f} +- {se:.3f} over {len(lob)} lobbies (4.5 is level)", flush=True)


if __name__ == "__main__":
    main()
