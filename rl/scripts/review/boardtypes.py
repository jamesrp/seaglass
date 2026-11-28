"""Which minion type a seat plays when the lobby offers it: does type school's lesson carry over?

Eight seats of the net (NET) play lobbies of `--tribes` minion types (default 5) with no buy limit.
For each minion type this counts the seats alive on turn 13 in a lobby that has the type ("offered")
and the ones whose board is of that type (`play.board_type`), with those boards' median total stats.
"none" counts the boards of no type.

    python boardtypes.py SEED0 N OUT.json [--tribes 5] [--curve standard|off] [--cap 80] [--workers 4]
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
    tribes = int(opt("--tribes", 5))
    opts = checkpoint_options(NET)
    curve = opt("--curve", "checkpoint")
    if curve != "checkpoint":
        opts.forced_curve = curriculum.parse_curve(curve)
    opts.max_steps_per_turn = int(opt("--cap", opts.max_steps_per_turn))
    opts.turn_stats = True
    config = LobbyConfig(tribes=tribes)
    player = Player(int(opt("--workers", 4)))
    specs = [GameSpec(seed=seed0 + i, controllers=["agent"] * 8) for i in range(n)]
    try:
        results = player.play(config, specs, {"agent": NET}, opts, record=(), seed=seed0)
    finally:
        player.close()
    offered, stats, every = {}, {}, []
    for r in results:
        types = sorted(config.make(r.seed).tribes)  # the lobby's types follow from its seed
        for seat in r.turn_stats:
            for s in seat:
                if s.turn != TURN:
                    continue
                every.append(s.stats)
                for t in types:
                    offered[t] = offered.get(t, 0) + 1
                stats.setdefault(s.type or "none", []).append(s.stats)
    table = {
        t: {"offered": offered.get(t, 0), "boards": len(stats.get(t, ())), "median_stats": st.median(stats[t]) if t in stats else 0}
        for t in sorted(set(offered) | set(stats))
    }
    result = {
        "net": NET,
        "lobbies": n,
        "seed": seed0,
        "tribes": tribes,
        "curve": curriculum.label(opts.forced_curve),
        "cap": opts.max_steps_per_turn,
        "turn": TURN,
        "seats": len(every),
        "median_stats": st.median(every) if every else 0,
        "types": table,
    }
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"{NET} | {n} lobbies of {tribes} minion type(s) | curve {result['curve']} | cap {result['cap']}")
    print(f"seats alive on turn {TURN}: {len(every)} | median board stats {result['median_stats']:,.0f}")
    print(f"{'type':<12} {'seats offered':>14} {'boards':>7} {'share when offered':>19} {'median stats':>13}")
    for t, r in table.items():
        share = f"{100 * r['boards'] / r['offered']:.1f}%" if r["offered"] else "-"
        print(f"{t:<12} {r['offered']:>14} {r['boards']:>7} {share:>19} {r['median_stats']:>13,.0f}")


if __name__ == "__main__":
    main()
