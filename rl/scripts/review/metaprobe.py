"""Fight the agents' end-of-turn boards (rebuilt from saved replays) against published human winner boards.

    python metaprobe.py MODE [CONTROLLER] [--scale 1.0]

The human boards are the "example winner board" of 13 current-patch comps (hspulse.net, November 2026),
with the stats as abbreviated on the page ("1K" -> 1000, a lower bound). Keywords beyond each card's own
are not on the page, nor are trinkets or Hero Powers, so the human boards are understated. --scale
multiplies every human stat (0.1 = a tenth of the published stats).

The ladder now does this for every rating (`seaglass_rl/yardstick.py`): this script is the review's
version, which reads the boards back from saved replays and also reports the win rate by comp.
"""

import collections
import glob
import json
import os
import statistics as st
import sys

import seaglass
from seaglass_rl import yardstick

# The folder the logs are in (`watch.py` writes them to REVIEW_DIR/games).
S = os.environ.get("REVIEW_DIR", "review")

# (name, turn, [(card, attack, health, golden)]): the boards of the ladder's human yardstick
# (`seaglass_rl/yardstick.py`, `seaglass_rl/data/human_boards/hspulse-2026-11.json`).
HUMAN = [
    (b["comp"], b["turn"], [(m["card"], m["attack"], m["health"], int(bool(m.get("golden")))) for m in b["minions"]])
    for b in yardstick.load("hspulse-2026-11")["boards"]
]


def slug(name):
    return name.lower().replace(" ", "_").replace("-", "_").replace("'", "_").replace(",", "")


def human_board(spec, scale):
    out = []
    for name, a, h, golden in spec:
        a2, h2 = max(1, int(a * scale)), max(1, int(h * scale))
        if name is None:
            s = f"{a2}/{h2} murloc t5"
        else:
            s = f"{a2}/{h2} card:{slug(name)}" + (" golden" if golden else "")
        out.append(seaglass.parse_unit(s))
    return out


def boards_from_replay(path, turns):
    """{turn: [(player, board units, tier)]} at the end of shopping (just before the turn's first combat)."""
    tl = seaglass.Timeline.load(path)
    n = len(tl)
    first = {}
    for i in range(n):
        fr = tl[i]
        if fr.kind in ("combat", "ghost_combat") and fr.turn in turns and fr.turn not in first:
            first[fr.turn] = i
    out = {}
    for turn, i in first.items():
        rows = []
        for p in range(tl.num_players):
            try:
                o = tl.player_at(i - 1, p).observe()
            except Exception:
                continue
            if o.health + o.armor <= 0 or not o.board:
                continue
            rows.append((p, list(o.board), o.tavern_tier))
        out[turn] = rows
    return out


def main():
    mode = sys.argv[1]
    ctrl = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    scale = float(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else 1.0
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 10**9
    turns = (10, 11, 12, 13, 14, 15, 16)
    refs = [(name, t, human_board(spec, scale)) for name, t, spec in HUMAN]
    wins = collections.defaultdict(list)  # turn -> per board mean win rate vs the 13 human boards
    per_comp = collections.defaultdict(list)
    files = sorted(glob.glob(f"{S}/games/{mode}_*.replay.pb"))[:limit]
    for f in files:
        meta = json.load(open(f.replace(".replay.pb", ".json")))
        for turn, rows in boards_from_replay(f, turns).items():
            for p, board, tier in rows:
                if ctrl and meta["controllers"][p] != ctrl:
                    continue
                tot = 0.0
                for k, (name, t, ref) in enumerate(refs):
                    d = seaglass.simulate_batch(board, ref, 1000 * turn + 17 * p + k, 4, hero_tier_a=tier, hero_tier_b=6)
                    w = d.a_win_rate + 0.5 * d.draw_rate
                    tot += w
                    if turn in (13, 14, 15):
                        per_comp[name].append(w)
                wins[turn].append(tot / len(refs))
    print(f"== {mode}/{ctrl} vs 13 human example winner boards (their stats x{scale}); {len(files)} lobbies")
    for turn in sorted(wins):
        w = wins[turn]
        print(f"  turn {turn}: {len(w):4d} boards, mean win rate {st.mean(w):.3f}, share of boards that beat any human board more than half the time: {sum(x > 0.5 / 13 for x in w) / len(w):.2f}")
    print("  by comp (turns 13-15): " + "; ".join(f"{k} {st.mean(v):.2f}" for k, v in per_comp.items()))


if __name__ == "__main__":
    main()
