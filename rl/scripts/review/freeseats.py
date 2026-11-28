"""Free seats against seats under type school's buy limit, in the same lobby.

Eight seats of the net (NET) play lobbies of `--tribes` minion types (default 1). Four of them are
under the buy limit (`PlayOptions.school_buys`, `own`: only minions of the lobby's types) and four
buy freely (`GameSpec.free_units`); which four alternates from lobby to lobby. For each group of
lobbies by type this prints both sides' average place, the median total stats of their boards on
turn 13 and how many of those boards are of one of the lobby's types.

    python freeseats.py SEED0 N OUT.json [--tribes 1] [--curve standard|off] [--cap 80] [--workers 4]

Where the limited seats place better, the net plays the type better than it chooses to.
"""

import json
import os
import statistics as st
import sys

from seaglass_rl import curriculum
from seaglass_rl.ladder import checkpoint_options
from seaglass_rl.play import GameSpec, LobbyConfig, Player

NET = os.environ.get("NET", "runs/capacity/small/latest.pt")
TURN = 13


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
    opts.school_buys = "own"
    config = LobbyConfig(tribes=tribes)
    player = Player(int(opt("--workers", 4)))
    specs = [
        GameSpec(seed=seed0 + i, controllers=["agent"] * 8, tribes=tribes, free_units=tuple(u for u in range(8) if (u + i) % 2 == 0))
        for i in range(n)
    ]
    try:
        results = player.play(config, specs, {"agent": NET}, opts, record=(), seed=seed0)
    finally:
        player.close()
    by_seed = {spec.seed: spec for spec in specs}  # the results do not come back in the specs' order
    groups = {}
    for r in results:
        types = sorted(config.make(r.seed).tribes)  # the lobby's types follow from its seed
        g = groups.setdefault("+".join(types) if tribes == 1 else "all", {"lobbies": 0, "free": ([], [], []), "limited": ([], [], [])})
        g["lobbies"] += 1
        for u in range(8):
            places, stats, own = g["free" if u in by_seed[r.seed].free_units else "limited"]
            places.append(r.placements[u])
            for s in r.turn_stats[u]:
                if s.turn == TURN:
                    stats.append(s.stats)
                    own.append(1.0 if s.type in types else 0.0)
    side = lambda v: {"place": st.mean(v[0]), "seats": len(v[1]), "median_stats": st.median(v[1]) if v[1] else 0, "own_type": st.mean(v[2]) if v[2] else 0.0}
    table = [{"types": k, "lobbies": g["lobbies"], "free": side(g["free"]), "limited": side(g["limited"])} for k, g in sorted(groups.items())]
    result = {
        "net": NET,
        "lobbies": n,
        "seed": seed0,
        "tribes": tribes,
        "curve": curriculum.label(opts.forced_curve),
        "cap": opts.max_steps_per_turn,
        "turn": TURN,
        "by_types": table,
    }
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"{NET} | {n} lobbies of {tribes} minion type(s) | curve {result['curve']} | cap {result['cap']} | four free seats, four under the limit")
    print(f"{'types':<12} lobbies   place: free limited   stats{TURN}: free limited   own type: free limited")
    for r in table:
        f, l = r["free"], r["limited"]
        print(
            f"{r['types']:<12} {r['lobbies']:>7} {f['place']:>13.2f} {l['place']:>7.2f} {f['median_stats']:>15,.0f} {l['median_stats']:>7,.0f} "
            f"{100 * f['own_type']:>14.0f}% {100 * l['own_type']:>6.0f}%"
        )


if __name__ == "__main__":
    main()
