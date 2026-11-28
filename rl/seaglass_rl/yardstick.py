"""The human yardstick: an agent's end-of-turn boards against boards that human players won with.

The ladder (`ladder.py`) and the probe (`probe.py`) rank an agent against the rule bots and against
its own peers. Both are relative: a field in which nobody builds a strong late board looks the same
from inside as one in which everybody does. The yardstick is the absolute mark. It fights each of the
agent's late boards against a fixed set of human winner boards and reports the win rate (a draw
counts half), by turn.

The boards are data (`data/human_boards/NAME.json`): one entry per board, with the turn it was taken
on and each minion's card, Attack, Health and whether it is golden. They are written by hand from a
published source or taken from parsed game logs. Nothing here is on the training path.

As in the probe, the fights are board-only (`seaglass.simulate_batch`): Hero Powers, trinkets and
other effects outside the board are not included.

`YardstickSpec.scales` multiplies every human stat. At the published stats (scale 1) a weak agent
wins almost nothing, and a number near zero can't show progress. A second, smaller scale puts the
same agent near the middle of the range, where a change is visible. The gate is the scale-1 number.

    python -m seaglass_rl.yardstick                 # list the boards and their total stats
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import seaglass

DATA = Path(__file__).resolve().parent / "data" / "human_boards"
# The human boards fight as a player on the top Tavern Tier.
HUMAN_TIER = 6


@dataclass(frozen=True)
class YardstickSpec:
    """Which human boards to fight, and how."""

    name: str = "hspulse-2026-11"  # data/human_boards/NAME.json
    # Each scale multiplies every human minion's Attack and Health; one win rate is reported per scale.
    scales: tuple[float, ...] = (1.0, 0.1)
    # The agent's boards are scored on these turns (the human boards are from turns 12 to 15).
    min_turn: int = 10
    max_turn: int = 16
    sims: int = 4  # fights per (board, human board) pair


def load(name: str) -> dict[str, Any]:
    """The data file of the named set."""
    with open(DATA / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def _slug(card: str) -> str:
    """The engine's card name in a unit spec (`seaglass.parse_unit`)."""
    return card.lower().replace(" ", "_").replace("-", "_").replace("'", "_").replace(",", "")


def unit(minion: dict[str, Any], scale: float = 1.0) -> Any:
    """One human minion as a `seaglass.Unit`, its stats multiplied by `scale` (at least 1 each)."""
    attack = max(1, int(minion["attack"] * scale))
    health = max(1, int(minion["health"] * scale))
    if minion.get("card") is None:
        # Not in the engine's catalog: a plain minion of its type and Tavern Tier.
        return seaglass.parse_unit(f"{attack}/{health} {minion.get('tribe', 'none')} t{minion.get('tier', 1)}")
    spec = f"{attack}/{health} card:{_slug(minion['card'])}"
    return seaglass.parse_unit(spec + (" golden" if minion.get("golden") else ""))


# name -> scale -> [board units], in the file's order. `seaglass.Unit` can't be pickled, so each
# worker process builds its own.
_CACHE: dict[tuple[str, float], list[list[Any]]] = {}


def boards(name: str, scale: float = 1.0) -> list[list[Any]]:
    """The named set's boards at `scale`, built once per process."""
    key = (name, float(scale))
    if key not in _CACHE:
        _CACHE[key] = [[unit(m, scale) for m in b["minions"]] for b in load(name)["boards"]]
    return _CACHE[key]


def score(spec: YardstickSpec, board: list, tier: int, turn: int, seed: int) -> tuple[float, ...] | None:
    """The board's mean score against every human board, in [0, 1], one value per scale of `spec`.
    None outside the spec's turns."""
    if not spec.min_turn <= turn <= spec.max_turn:
        return None
    out = []
    for scale in spec.scales:
        refs = boards(spec.name, scale)
        total = 0.0
        for i, ref in enumerate(refs):
            d = seaglass.simulate_batch(board, ref, seed + i, spec.sims, hero_tier_a=tier, hero_tier_b=HUMAN_TIER)
            total += d.a_win_rate + 0.5 * d.draw_rate
        out.append(total / len(refs))
    return tuple(out)


def summary(name: str) -> list[dict[str, Any]]:
    """Each board of the set: its comp, turn and total stats (the sum of Attack and Health)."""
    return [
        {
            "comp": b.get("comp"),
            "turn": b.get("turn"),
            "minions": len(b["minions"]),
            "stats": sum(m["attack"] + m["health"] for m in b["minions"]),
        }
        for b in load(name)["boards"]
    ]


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=YardstickSpec().name)
    args = parser.parse_args(argv)
    data = load(args.name)
    boards(args.name)  # every card resolves in this engine build, or this raises
    print(f"{data['name']} | patch {data.get('patch', '?')} | {data.get('source', '?')} | {len(data['boards'])} boards")
    for row in summary(args.name):
        print(f"  turn {row['turn']:>2} | {row['stats']:>7,} total stats | {row['minions']} minions | {row['comp']}")


if __name__ == "__main__":
    main()
