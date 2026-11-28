"""What it costs a seat to commit to one minion type in a five-type lobby.

Eight seats of the net (NET) play lobbies of `--tribes` minion types (default 5) that offer TYPE.
From turn `--from` (default 1) two of the seats are limited in what they buy; the other six buy
freely. `--limit`:

    both    only TYPE's minions (and minions of every type), and no Tavern spell   (default)
    type    only TYPE's minions (and minions of every type); Tavern spells are allowed
    spells  every minion; no Tavern spell

It prints both sides' average place, the median total stats of their boards on turn 13 and how many
of those boards are of TYPE, for all the lobbies and for the ones with and without Mechs.

    python committed.py TYPE SEED0 N OUT.json [--from 1] [--limit both|type|spells] [--tribes 5] [--curve standard|off] [--cap 80]

The limit is a what-if of this script, not a training option: it stands in for `play.limit_buys`, so
the games run in this process (about a minute for 60 lobbies).
"""

import json
import os
import statistics as st
import sys

from seaglass_rl import checkpoint, curriculum, play
from seaglass_rl.ladder import checkpoint_options
from seaglass_rl.play import GameSpec, LobbyConfig

NET = os.environ.get("NET", "runs/capacity/small/latest.pt")
TURN = 13
LIMITS = ("both", "type", "spells")


def main():
    kind, seed0, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    args = sys.argv[5:]
    opt = lambda name, default: args[args.index(name) + 1] if name in args else default
    start, limit, tribes = int(opt("--from", 1)), opt("--limit", "both"), int(opt("--tribes", 5))
    if limit not in LIMITS:
        raise SystemExit(f"--limit: one of {', '.join(LIMITS)}")

    def commit(mask, seat, obs, _limit):
        """`play.limit_buys` for a committed seat: from turn `start`, mask the buys `limit` rules out."""
        buys = [s for s, i in enumerate(play.BUY_INDICES) if mask[i]]
        if not buys or seat.game.phase != "turn" or seat.game.turn < start:
            return
        shop = (obs if obs is not None else seat.observe()).shop
        for s in buys:
            u = shop[s]
            spell = u.is_spell and limit in ("both", "spells")
            other = not u.is_spell and limit in ("both", "type") and u.tribe != "all" and kind not in u.tribes
            if spell or other:
                mask[play.BUY_INDICES[s]] = False

    play.limit_buys = commit
    model, _ = checkpoint.load(NET)
    opts = checkpoint_options(NET)
    curve = opt("--curve", "checkpoint")
    if curve != "checkpoint":
        opts.forced_curve = curriculum.parse_curve(curve)
    opts.max_steps_per_turn = int(opt("--cap", opts.max_steps_per_turn))
    opts.turn_stats = True
    opts.school_buys = "own"  # the committed seats are the ones `GameSpec.free_units` leaves out
    config = LobbyConfig(tribes=tribes)
    seeds, seed = [], seed0
    while len(seeds) < n:  # the lobby's types follow from its seed
        if kind in config.make(seed).tribes:
            seeds.append(seed)
        seed += 1
    rows = []
    for i, seed in enumerate(seeds):
        committed = (i % 8, (i + 4) % 8)
        spec = GameSpec(seed=seed, controllers=["agent"] * 8, tribes=tribes, free_units=tuple(u for u in range(8) if u not in committed))
        (r,) = play.play_games(config, [spec], {"agent": model}, opts, record=())
        mech = "mech" in config.make(seed).tribes
        for u in range(8):
            on_turn = [s for s in r.turn_stats[u] if s.turn == TURN]
            rows.append((u in committed, mech, r.placements[u], on_turn[0].stats if on_turn else None, on_turn[0].type if on_turn else None))

    def side(group):
        alive = [x for x in group if x[3] is not None]
        return {
            "place": st.mean(x[2] for x in group) if group else 0.0,
            "seats": len(alive),
            "median_stats": st.median(x[3] for x in alive) if alive else 0,
            "own_type": st.mean(1.0 if x[4] == kind else 0.0 for x in alive) if alive else 0.0,
        }

    def group(name, keep):
        mine = [x for x in rows if keep(x)]
        return {"lobbies_of": name, "lobbies": len(mine) // 8, "committed": side([x for x in mine if x[0]]), "free": side([x for x in mine if not x[0]])}

    table = [group("all", lambda x: True), group("no mech", lambda x: not x[1]), group("with mech", lambda x: x[1])]
    result = {
        "net": NET,
        "type": kind,
        "lobbies": n,
        "seed": seed0,
        "tribes": tribes,
        "from_turn": start,
        "limit": limit,
        "curve": curriculum.label(opts.forced_curve),
        "cap": opts.max_steps_per_turn,
        "turn": TURN,
        "groups": table,
    }
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"{NET} | {n} lobbies of {tribes} minion types that offer {kind} | curve {result['curve']} | cap {result['cap']}")
    print(f"two seats committed to {kind} from turn {start} (limit: {limit}), six free")
    print(f"{'lobbies':<10} {'count':>5}   place: committed free   stats{TURN}: committed free   {kind} boards: committed free")
    for r in table:
        c, f = r["committed"], r["free"]
        print(
            f"{r['lobbies_of']:<10} {r['lobbies']:>5}   {c['place']:>16.2f} {f['place']:>4.2f}   "
            f"{c['median_stats']:>18,.0f} {f['median_stats']:>5,.0f}   {100 * c['own_type']:>{len(kind) + 16}.0f}% {100 * f['own_type']:>3.0f}%"
        )


if __name__ == "__main__":
    main()
