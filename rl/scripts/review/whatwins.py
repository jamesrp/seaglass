"""What wins a place among the net's own seats: the learning signal of self-play.

Eight seats of the net (NET) play each lobby. For one turn (`--turn`, default 11), every seat still
alive is grouped by what its board holds when the shopping ends, and each group's average final
place is printed: by the number of Tier 5 and 6 minions, by the board's total stats, and by the
board's minion type. If a group does not place better, self-play has no reason to move toward it.

    python whatwins.py SEED0 N OUT.json [--curve standard|off] [--cap 80] [--turn 11] [--workers 4]
"""

import json
import os
import statistics as st
import sys

from seaglass_rl import curriculum
from seaglass_rl.ladder import checkpoint_options
from seaglass_rl.play import GameSpec, LobbyConfig, Player

NET = os.environ.get("NET", "runs/capacity/small/latest.pt")


def corr(xs, ys):
    """Pearson correlation of the ranks (Spearman)."""

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    return st.correlation(ranks(xs), ranks(ys)) if len(set(xs)) > 1 and len(set(ys)) > 1 else float("nan")


def main():
    seed0, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    args = sys.argv[4:]
    opt = lambda name, default: args[args.index(name) + 1] if name in args else default
    turn = int(opt("--turn", 11))
    opts = checkpoint_options(NET)
    curve = opt("--curve", "checkpoint")
    if curve != "checkpoint":
        opts.forced_curve = curriculum.parse_curve(curve)
    opts.max_steps_per_turn = int(opt("--cap", opts.max_steps_per_turn))
    opts.turn_stats = True
    player = Player(int(opt("--workers", 4)))
    specs = [GameSpec(seed=seed0 + i, controllers=["agent"] * 8) for i in range(n)]
    try:
        results = player.play(LobbyConfig(), specs, {"agent": NET}, opts, record=(), seed=seed0)
    finally:
        player.close()
    seats = []  # every seat alive on the turn: its board then, and its final place
    for r in results:
        for u, stats in enumerate(r.turn_stats):
            s = next((s for s in stats if s.turn == turn), None)
            if s is not None:
                seats.append({"high": s.high, "stats": s.stats, "type": s.type or "none", "place": r.placements[u], "tier": s.tier})
    place = [s["place"] for s in seats]
    result = {
        "net": NET,
        "lobbies": n,
        "turns": st.mean(r.turns for r in results),
        "seed": seed0,
        "curve": curriculum.label(opts.forced_curve),
        "cap": opts.max_steps_per_turn,
        "turn": turn,
        "seats": len(seats),
        "mean_place": st.mean(place),
        "rank_corr_high_place": corr([s["high"] for s in seats], place),
        "rank_corr_stats_place": corr([s["stats"] for s in seats], place),
    }

    def groups(key, label):
        rows = {}
        for s in seats:
            rows.setdefault(key(s), []).append(s)
        table = []
        for k in sorted(rows, key=lambda k: (isinstance(k, str), k)):
            g = rows[k]
            table.append({label: k, "seats": len(g), "place": st.mean(x["place"] for x in g), "stats": st.median(x["stats"] for x in g), "mean_high": st.mean(x["high"] for x in g)})
        return table

    result["by_high"] = groups(lambda s: min(s["high"], 4), "high")
    cuts = st.quantiles([s["stats"] for s in seats], n=4)
    result["by_stats_quarter"] = groups(lambda s: sum(s["stats"] > c for c in cuts) + 1, "quarter")
    result["by_type"] = sorted(groups(lambda s: s["type"], "type"), key=lambda r: -r["seats"])
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"{NET} | {n} lobbies | curve {result['curve']} | turn {turn} | {len(seats)} of {8 * n} seats alive, average place {result['mean_place']:.2f} | a lobby lasts {result['turns']:.1f} turns")
    print(f"rank correlation with the final place (negative is better): Tier 5 and 6 minions {result['rank_corr_high_place']:+.3f}, total stats {result['rank_corr_stats_place']:+.3f}")
    print("Tier 5 and 6 minions on the board (4 is 4 or more):")
    for r in result["by_high"]:
        print(f"    {r['high']}: place {r['place']:.2f} | {r['seats']:>5} seats | median stats {r['stats']:,.0f}")
    print("total stats, by quarter of all seats (1 is the lowest):")
    for r in result["by_stats_quarter"]:
        print(f"    {r['quarter']}: place {r['place']:.2f} | {r['seats']:>5} seats | median stats {r['stats']:,.0f} | Tier 5 and 6 minions {r['mean_high']:.2f}")
    print("the board's minion type:")
    for r in result["by_type"]:
        if r["seats"] >= 20:
            print(f"    {r['type']:<11} place {r['place']:.2f} | {r['seats']:>5} seats | median stats {r['stats']:,.0f} | Tier 5 and 6 minions {r['mean_high']:.2f}")


if __name__ == "__main__":
    main()
