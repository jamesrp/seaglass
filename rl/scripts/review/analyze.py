"""Aggregate the watch logs: leveling curve, rolls, board make-up by turn, per controller."""

import collections
import glob
import json
import os
import statistics as st
import sys

# The folder the logs are in (`watch.py` writes them to REVIEW_DIR/games).
S = os.environ.get("REVIEW_DIR", "review")


def load(mode):
    out = []
    for f in sorted(glob.glob(f"{S}/games/{mode}_*.json")):
        out.append(json.load(open(f)))
    return out


def seats(games, controller=None):
    for g in games:
        for p, pl in g["players"].items():
            c = g["controllers"][int(p)]
            if controller is None or c == controller:
                yield g, int(p), c, pl


def curve(games, controller=None, maxturn=18):
    rows = collections.defaultdict(list)
    for g, p, c, pl in seats(games, controller):
        for t in pl["turns"]:
            rows[t["turn"]].append(t)
    print(f"{'turn':>4} {'n':>5} {'tier':>5} {'%T>=4':>6} {'%T>=5':>6} {'%T6':>5} {'gold':>5} {'hp':>5} {'brd':>4} {'rolls':>5} {'buys':>5} {'sells':>5} {'lvl':>4} {'DD':>4} {'mTier':>5} {'%m>=5':>6} {'gold#':>5} {'sumstat':>8} {'maxstat':>8}")
    res = {}
    for turn in sorted(rows):
        if turn > maxturn:
            break
        ts = rows[turn]
        n = len(ts)
        tier = st.mean(t["tier"] for t in ts)
        f4 = sum(t["tier"] >= 4 for t in ts) / n
        f5 = sum(t["tier"] >= 5 for t in ts) / n
        f6 = sum(t["tier"] >= 6 for t in ts) / n
        gold = st.mean(t["gold_left"] for t in ts)
        hp = st.mean(t["health"] for t in ts)
        brd = st.mean(len(t["board"]) for t in ts)

        def cnt(t, pre):
            return sum(a.startswith(pre) for a in t["actions"])

        rolls = st.mean(cnt(t, "roll") for t in ts)
        buys = st.mean(cnt(t, "buy") for t in ts)
        sells = st.mean(cnt(t, "sell") for t in ts)
        lvl = st.mean(cnt(t, "LEVEL") for t in ts)
        dd = st.mean(cnt(t, "dark_discovery") for t in ts)
        mins = [u for t in ts for u in t["board"]]
        mtier = st.mean(u["tier"] for u in mins) if mins else 0
        m5 = sum(u["tier"] >= 5 for u in mins) / max(1, len(mins))
        golden = st.mean(sum(1 for u in t["board"] if u.get("golden")) for t in ts)
        sumstat = st.median(sum(u["atk"] + u["hp"] for u in t["board"]) for t in ts)
        maxstat = st.median(max([u["atk"] + u["hp"] for u in t["board"]] or [0]) for t in ts)
        res[turn] = dict(n=n, tier=tier, f4=f4, f5=f5, f6=f6, rolls=rolls, sumstat=sumstat, mtier=mtier, m5=m5)
        print(f"{turn:>4} {n:>5} {tier:>5.2f} {f4:>6.0%} {f5:>6.0%} {f6:>5.0%} {gold:>5.1f} {hp:>5.1f} {brd:>4.1f} {rolls:>5.1f} {buys:>5.1f} {sells:>5.1f} {lvl:>4.2f} {dd:>4.2f} {mtier:>5.2f} {m5:>6.0%} {golden:>5.2f} {sumstat:>8.0f} {maxstat:>8.0f}")
    return res


def level_turns(games, controller=None):
    """Turn on which each seat first reached each tier."""
    first = collections.defaultdict(list)
    never = collections.Counter()
    n = 0
    for g, p, c, pl in seats(games, controller):
        n += 1
        seen = {}
        for t in pl["turns"]:
            for k in range(2, t["tier"] + 1):
                seen.setdefault(k, t["turn"])
        for k in range(2, 7):
            if k in seen:
                first[k].append(seen[k])
            else:
                never[k] += 1
    print("first turn at tier (median, mean; share of seats that ever got there):")
    for k in range(2, 7):
        if first[k]:
            print(f"  T{k}: median {st.median(first[k]):.0f}, mean {st.mean(first[k]):.1f}, reached {len(first[k]) / n:.0%}")
        else:
            print(f"  T{k}: never")


def final_boards(games, controller=None, top=None, k=30):
    cards = collections.Counter()
    tribes = collections.Counter()
    nseats = 0
    purity = []
    for g, p, c, pl in seats(games, controller):
        place = g["placements"][p]
        if top and place > top:
            continue
        if not pl["turns"]:
            continue
        b = pl["turns"][-1]["board"]
        nseats += 1
        for u in b:
            cards[(u["name"], u["tier"])] += 1
        tc = collections.Counter()
        for u in b:
            for tr in u["tribe"].split("_"):
                tc[tr] += 1
        if b:
            real = {t: n for t, n in tc.items() if t not in ("none", "all")}
            best = max(real.values()) if real else 0
            purity.append((best + tc.get("all", 0)) / len(b))
            if real:
                tribes[max(real, key=real.get)] += 1
    print(f"final boards: {nseats} seats; mean share of board in the main type {st.mean(purity):.0%}")
    print("  most common cards (name, tier, share of boards with >=1 copy is approximated by count/seats):")
    print("  " + "; ".join(f"{n} T{t} {c / nseats:.2f}" for (n, t), c in cards.most_common(k)))
    return cards, nseats


if __name__ == "__main__":
    mode = sys.argv[1]
    ctrl = sys.argv[2] if len(sys.argv) > 2 else None
    games = load(mode)
    print(f"== {mode} / {ctrl}: {len(games)} lobbies; mean length {st.mean(g['turns'] for g in games):.1f} turns")
    if ctrl:
        pl = [g["placements"][p] for g, p, c, _ in seats(games, ctrl)]
        print(f"   avg place {st.mean(pl):.2f} over {len(pl)} seats")
    curve(games, ctrl)
    level_turns(games, ctrl)
    final_boards(games, ctrl)
    print("  -- winners only")
    final_boards(games, ctrl, top=1)
