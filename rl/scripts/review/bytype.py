"""What the net builds when the lobby has few minion types: the type school's measure.

Eight seats of the net (NET) play lobbies of `--tribes` minion types (default 1). The lobbies are
grouped by their types, and each group's boards are printed: the lobby's length, the median total
stats on turns 11 and 13, the Tier 5 and 6 minions and the win rate against the human winner boards
on turn 13, and how many boards on turn 13 are of one of the lobby's types.

    python bytype.py SEED0 N OUT.json [--tribes 1] [--curve standard|off] [--cap 80] [--workers 4]
"""

import json
import os
import statistics as st
import sys

from seaglass_rl import curriculum
from seaglass_rl.ladder import checkpoint_options
from seaglass_rl.play import GameSpec, LobbyConfig, Player
from seaglass_rl.yardstick import YardstickSpec

NET = os.environ.get("NET", "runs/capacity/small/latest.pt")


def main():
    seed0, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    args = sys.argv[4:]
    opt = lambda name, default: args[args.index(name) + 1] if name in args else default
    tribes = int(opt("--tribes", 1))
    opts = checkpoint_options(NET)
    curve = opt("--curve", "checkpoint")
    if curve != "checkpoint":
        opts.forced_curve = curriculum.parse_curve(curve)
    opts.max_steps_per_turn = int(opt("--cap", opts.max_steps_per_turn))
    opts.turn_stats = True
    opts.yardstick = YardstickSpec()
    config = LobbyConfig(tribes=tribes)
    player = Player(int(opt("--workers", 4)))
    specs = [GameSpec(seed=seed0 + i, controllers=["agent"] * 8, probe_units=tuple(range(8))) for i in range(n)]
    try:
        results = player.play(config, specs, {"agent": NET}, opts, record=(), seed=seed0)
    finally:
        player.close()
    groups = {}
    for r in results:
        types = sorted(config.make(r.seed).tribes)  # the lobby's types follow from its seed
        g = groups.setdefault("+".join(types), {"turns": [], "stats11": [], "stats13": [], "high": [], "human": [], "h10": [], "own": []})
        g["turns"].append(r.turns)
        for stats in r.turn_stats:
            for s in stats:
                if s.turn == 11:
                    g["stats11"].append(s.stats)
                if s.turn == 13:
                    g["stats13"].append(s.stats)
                    g["high"].append(s.high)
                    g["own"].append(1.0 if s.type in types else 0.0)
                    if s.yardstick is not None:
                        g["human"].append(s.yardstick[0])
                        g["h10"].append(s.yardstick[1])
    mean = lambda v: st.mean(v) if v else float("nan")
    table = [
        {
            "types": k,
            "lobbies": len(g["turns"]),
            "turns": mean(g["turns"]),
            "stats11": st.median(g["stats11"]) if g["stats11"] else 0,
            "stats13": st.median(g["stats13"]) if g["stats13"] else 0,
            "high13": mean(g["high"]),
            "human13": mean(g["human"]),
            "human10pct13": mean(g["h10"]),
            "own_type13": mean(g["own"]),
            "seats13": len(g["stats13"]),
        }
        for k, g in sorted(groups.items())
    ]
    over = {key: st.mean(row[key] for row in table) for key in ("turns", "stats11", "stats13", "high13", "human13", "human10pct13", "own_type13")}
    result = {
        "net": NET,
        "lobbies": n,
        "seed": seed0,
        "tribes": tribes,
        "curve": curriculum.label(opts.forced_curve),
        "cap": opts.max_steps_per_turn,
        "by_types": table,
        "mean_over_types": over,
    }
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"{NET} | {n} lobbies of {tribes} minion type(s) | curve {result['curve']} | cap {result['cap']}")
    print(f"{'types':<22} lobbies turns stats11 stats13   t5+  human  h@10%  own type")
    for r in [*table, {"types": "mean over the types", "lobbies": n, **over}]:
        print(
            f"{r['types']:<22} {r['lobbies']:>7} {r['turns']:5.1f} {r['stats11']:7,.0f} {r['stats13']:7,.0f} {r['high13']:5.1f} "
            f"{100 * r['human13']:5.1f}% {100 * r['human10pct13']:5.1f}% {100 * r['own_type13']:8.0f}%"
        )
    print("stats: the median board's total Attack + Health on turns 11 and 13 | t5+, human, h@10%, own type: boards on turn 13")


if __name__ == "__main__":
    main()
