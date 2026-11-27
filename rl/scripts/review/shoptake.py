"""What a net buys from the shops it is shown.

A shop "view" is a turn's first shop or the shop after a roll, counted when the seat has the Gold
to buy from it (3 or more). For every view this counts the minions offered, by the seat's Tavern
Tier and the minion's Tier, and the ones the seat buys before its next roll or the end of the turn.

    python shoptake.py SEED0 N OUT.json [--curve standard|off] [--cap 80] [--from-turn 9]

All eight seats are the net (NET). `--curve` forces every seat onto a leveling curve
(`curriculum.py`), as in endgame school; the default is the checkpoint's own.
"""

import json
import os
import sys
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch

torch.set_num_threads(1)

from seaglass_rl import checkpoint, curriculum, play
from seaglass_rl.actions import action_from_index
from seaglass_rl.ladder import checkpoint_options
from seaglass_rl.play import GameSpec, LobbyConfig

NET = os.environ.get("NET", "runs/capacity/small/latest.pt")

OFFER: Counter = Counter()  # (seat's Tavern Tier, minion's Tier) -> minions offered
BUY: Counter = Counter()  # the same key -> minions bought
CARD_OFFER: Counter = Counter()  # (minion's Tier, name) -> offered, on FROM_TURN and later
CARD_BUY: Counter = Counter()
VIEW: dict = {}  # (game, player) -> the view the seat is looking at
FROM_TURN = 9

_orig_apply = play._apply


def _apply(game, p, st, idx, note, kind=None):
    if kind is None:  # the seat's own action, not the environment's
        a = action_from_index(idx)
        o = game.observe(p)
        key = (id(game), p)
        v = VIEW.get(key)
        if v is None or v["turn"] != game.turn or v["rolled"]:
            v = VIEW[key] = {"turn": game.turn, "rolled": False, "live": o.gold >= 3}
            if v["live"]:
                for u in o.shop:
                    if not u.is_spell:
                        OFFER[(o.tavern_tier, u.tavern_tier)] += 1
                        if game.turn >= FROM_TURN:
                            CARD_OFFER[(u.tavern_tier, u.name)] += 1
        if a.kind == "buy" and v["live"]:
            u = o.shop[a.args[0]]
            if not u.is_spell:
                BUY[(o.tavern_tier, u.tavern_tier)] += 1
                if game.turn >= FROM_TURN:
                    CARD_BUY[(u.tavern_tier, u.name)] += 1
        elif a.kind == "refresh":
            v["rolled"] = True
    return _orig_apply(game, p, st, idx, note, kind)


play._apply = _apply


def main():
    global FROM_TURN
    seed0, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    args = sys.argv[4:]
    opt = lambda name, default: args[args.index(name) + 1] if name in args else default
    FROM_TURN = int(opt("--from-turn", FROM_TURN))
    model, _ = checkpoint.load(NET)
    opts = checkpoint_options(NET)
    curve = opt("--curve", "checkpoint")
    if curve != "checkpoint":
        opts.forced_curve = curriculum.parse_curve(curve)
    opts.max_steps_per_turn = int(opt("--cap", opts.max_steps_per_turn))
    cfg = LobbyConfig()
    for i in range(n):
        VIEW.clear()
        play.play_games(cfg, [GameSpec(seed=seed0 + i, controllers=["agent"] * 8)], {"agent": model}, opts, record=())
    tiers = range(1, 7)
    table = {
        str(tt): {str(mt): {"offered": OFFER[(tt, mt)], "bought": BUY[(tt, mt)]} for mt in tiers if OFFER[(tt, mt)]}
        for tt in tiers
    }
    cards = [
        {"tier": mt, "name": name, "offered": k, "bought": CARD_BUY[(mt, name)]}
        for (mt, name), k in sorted(CARD_OFFER.items(), key=lambda kv: -kv[1])
    ]
    result = {
        "net": NET,
        "lobbies": n,
        "seed": seed0,
        "curve": curriculum.label(opts.forced_curve),
        "cap": opts.max_steps_per_turn,
        "from_turn": FROM_TURN,
        "by_tier": table,
        "cards": cards,
    }
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"{NET} | {n} lobbies | curve {result['curve']} | cap {result['cap']}")
    print("take rate (bought / offered) by the seat's Tavern Tier (rows) and the minion's Tier (columns)")
    print("tavern " + " ".join(f"{'T' + str(mt):>12}" for mt in tiers))
    for tt in tiers:
        cells = []
        for mt in tiers:
            k = OFFER[(tt, mt)]
            cells.append(f"{100 * BUY[(tt, mt)] / k:5.1f}% {k:>5}" if k >= 30 else f"{'-':>12}")
        print(f"{tt:>6} " + " ".join(cells))
    for group, label in (((5, 6), "Tier 5 and 6"), ((1, 2, 3, 4), "Tier 1 to 4")):
        rows = [c for c in cards if c["tier"] in group and c["offered"] >= 40]
        rows.sort(key=lambda c: -c["bought"] / c["offered"])
        print(f"{label} minions offered on turn {FROM_TURN} and later, by take rate (40+ offers):")
        for c in rows[:12] + ([{"name": "..."}] if len(rows) > 18 else []) + (rows[-6:] if len(rows) > 18 else rows[12:]):
            if "tier" not in c:
                print("    ...")
                continue
            print(f"    {100 * c['bought'] / c['offered']:5.1f}%  {c['offered']:>5}  T{c['tier']} {c['name']}")


if __name__ == "__main__":
    main()
