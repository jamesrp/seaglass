"""Training scaffolds: the forced leveling curve ("endgame school") and lobbies of fewer minion
types ("type school").

Self-play alone settled on Tavern Tiers 2 and 3 (`docs/review-2026-11-27.md`): leveling costs tempo
now, its payoff is several turns and several specific cards away, and nobody in the pool shows the
learner a board that scales. So the learner never visits the late game and never learns to play it.

The forced curve puts it there. A curve is a list of `(turn, Tavern Tier)` steps. On a forced seat,
whenever the seat is below the curve's Tier for the turn and the Tavern upgrade is legal, the
environment takes the upgrade (`play.PlayOptions.forced_curve`). It is an environment event: it is
not one of the policy's decisions and is not recorded in the trajectory, so PPO never trains on it.
With every model seat forced, the cost is the same for all of them and nobody is punished for it.

This is a human-written play rule, so it is scaffolding in the same sense as the rule-bot warm
start: `train_lobby --pure` refuses it, and the stage that uses it ends by removing it
(`--force-share` below 1, then no curve). A network counts as leveling only when it does so with no
curve.

The type school is the second scaffold. On the curve the learner reached Tier 6 and went on buying
the ten high-tier cards it knew, in three of the ten minion types (`docs/neural-opponents-sprint.md`,
Stage 6 Progress). A lobby of five types always offers it those. A lobby of one type does not: its
shop holds that type's minions and the typeless ones, so every seat has to make that type work, and
the seat that does wins the lobby. A mix is a list of `(minion types, share)`: that share of the
training lobbies is played with that many types (`GameSpec.tribes`), the rest as the lobby config
says. It changes which lobbies are played and no decision inside them, and the evaluation is not
changed by it.

A one-type shop alone did not do it: the network rolled past the lobby's minions and bought Tavern
spells and typeless minions for the board it already had. The buy limit (`PlayOptions.school_buys`,
`train_lobby --school-buys own`) closes that way out: a model seat of a school lobby can buy only
minions of the lobby's types, so no Tavern spell and no typeless minion. Every seat of the lobby is
limited alike, and the seat that makes the type's cards work wins it. The limit masks Buy actions,
so the policy still decides everything it does and PPO trains on those decisions as usual. It is a
hand-written rule on what a seat buys, so `train_lobby --pure` refuses it, and no evaluation has it.
"""

from __future__ import annotations

Curve = tuple[tuple[int, int], ...]

# name -> ((turn, Tavern Tier), ...): from that turn on, the seat should be on that Tier or higher.
CURVES: dict[str, Curve] = {
    # The standard curve of the leveling guides to Tier 4; Tiers 5 and 6 two turns apart after it.
    "standard": ((2, 2), (5, 3), (7, 4), (9, 5), (11, 6)),
    "fast": ((2, 2), (4, 3), (6, 4), (8, 5), (10, 6)),
    "slow": ((2, 2), (5, 3), (8, 4), (10, 5), (12, 6)),
    # The standard curve, stopping at Tier 4 or Tier 5.
    "to4": ((2, 2), (5, 3), (7, 4)),
    "to5": ((2, 2), (5, 3), (7, 4), (9, 5)),
    # The seat's own early game, then late levels only.
    "late": ((9, 4), (11, 5), (13, 6)),
}


def parse_curve(text: str | None) -> Curve | None:
    """A curve from its name in `CURVES`, or from `"turn:tier,turn:tier,..."`. None, "" and "off"
    are no curve."""
    if text is None or text in ("", "off", "none"):
        return None
    if text in CURVES:
        return CURVES[text]
    try:
        steps = tuple(sorted((int(a), int(b)) for a, b in (part.split(":") for part in text.split(","))))
    except ValueError:
        raise ValueError(f"not a curve: {text!r} (a name from {', '.join(CURVES)}, or turn:tier,turn:tier,...)")
    if not steps or any(turn < 1 or not 2 <= tier <= 6 for turn, tier in steps):
        raise ValueError(f"not a curve: {text!r} (turns start at 1, Tavern Tiers are 2..6)")
    return steps


TypeMix = tuple[tuple[int, float], ...]


def parse_type_mix(text: str | None) -> TypeMix:
    """A type school mix from `"types:share,types:share,..."`, for example `"1:0.5"` (half the
    lobbies have one minion type) or `"1:0.4,2:0.2"`. None, "" and "off" are no mix."""
    if text is None or text in ("", "off", "none"):
        return ()
    try:
        mix = tuple((int(a), float(b)) for a, b in (part.split(":") for part in text.split(",")))
    except ValueError:
        raise ValueError(f"not a type mix: {text!r} (types:share,types:share,..., for example 1:0.5)")
    if any(not 1 <= n <= 10 or not 0.0 < share <= 1.0 for n, share in mix) or sum(share for _, share in mix) > 1.0 + 1e-9:
        raise ValueError(f"not a type mix: {text!r} (1 to 10 minion types, and shares above 0 that add up to 1 at most)")
    if len({n for n, _ in mix}) != len(mix):
        raise ValueError(f"not a type mix: {text!r} (a number of minion types is listed twice)")
    return mix


def draw_types(mix: TypeMix, x: float) -> int | None:
    """The minion types of a lobby whose draw is `x` (uniform in 0..1): the mix's entry whose share
    `x` falls in, or None (the lobby config's own) past the last one."""
    for n, share in mix:
        if x < share:
            return n
        x -= share
    return None


def target_tier(curve: Curve, turn: int) -> int:
    """The Tavern Tier the curve asks for on `turn` (1 before its first step)."""
    target = 1
    for at, tier in curve:
        if turn >= at:
            target = max(target, tier)
    return target


def label(curve: Curve | None) -> str:
    """The curve's name in `CURVES`, or its steps."""
    if curve is None:
        return "off"
    for name, steps in CURVES.items():
        if steps == tuple(curve):
            return name
    return ",".join(f"{turn}:{tier}" for turn, tier in curve)
