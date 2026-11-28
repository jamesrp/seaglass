"""Play lobbies and log every seat's turns in detail (boards, tiers, actions) for qualitative review.

    python watch.py MODE SEED0 N OUT_DIR [--search end|next_start_self|none]

MODE:
  allsearch  all 8 seats are the net with search
  mixed      2 seats net+search ("agent"), 6 seats the same net without search ("rival")
  slow       the slow ladder's field: 2 agent seats (+search), sovereign, 2x 9kgamerbot, 2x tempo, heuristic
  selfplain  all 8 seats the net without search
  bots       4x 9kgamerbot, 4x 9kgamerbot-tempo  (reference)
"""

import json
import os
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch

torch.set_num_threads(1)

import seaglass
from seaglass_rl import play
from seaglass_rl.actions import action_from_index
from seaglass_rl.ladder import checkpoint_options
from seaglass_rl.play import GameSpec, LobbyConfig
from seaglass_rl.probe import ProbeSpec
from seaglass_rl.search import SearchSpec
from seaglass_rl import checkpoint

# The checkpoint that plays the "agent" and "rival" seats.
NET = os.environ.get("NET", "runs/capacity/small/latest.pt")

HEROES = {i: n for i, n, _ in seaglass.heroes_table()}
TRINKETS = {r[0]: r[1] for r in seaglass.trinkets_table()}
GIFTS = {r[0]: r[1] for r in seaglass.dark_gifts_table()}


def unit_dict(u):
    d = {
        "name": u.name,
        "tier": u.tavern_tier,
        "atk": u.attack,
        "hp": u.health,
        "tribe": u.tribe,
    }
    if u.is_golden:
        d["golden"] = True
    kw = [k for k in ("divine_shield", "reborn", "taunt", "venomous", "windfury", "stealth") if getattr(u, k)]
    if kw:
        d["kw"] = kw
    if u.dark_gifts:
        d["gifts"] = [GIFTS.get(g, str(g)) for g in u.dark_gifts]
    if u.is_spell:
        d["spell"] = True
    return d


LOG = {}  # id(game) -> {player -> [action strings this turn]}
GAMES = {}  # id(game) -> record

_orig_apply = play._apply


def _describe(game, p, idx):
    a = action_from_index(idx)
    kind, args = a.kind, a.args
    try:
        o = game.observe(p)
    except Exception:
        return kind
    try:
        if kind == "buy":
            return f"buy {o.shop[args[0]].name}"
        if kind == "sell":
            u = o.board[args[0]]
            return f"sell {u.name}{'*' if u.is_golden else ''}"
        if kind in ("play", "play_on_shop"):
            u = o.hand[args[0]]
            return f"play {u.name}{'*' if u.is_golden else ''}"
        if kind == "choose_discover":
            opts = [u.name for u in o.discover_pending] if o.discover_pending else []
            pick = opts[args[0]] if args[0] is not None and args[0] < len(opts) else "?"
            return f"discover {pick} (of {', '.join(opts)})"
        if kind == "choose_trinket":
            opts = [TRINKETS.get(t, str(t)) for t in o.trinket_options]
            return f"trinket {opts[args[0]]} (of {', '.join(opts)})"
        if kind == "choose_hero":
            opts = [HEROES.get(h, str(h)) for h in o.hero_options]
            return f"hero {opts[args[0]]} (of {', '.join(opts)})"
        if kind == "upgrade_tavern":
            return f"LEVEL->{o.tavern_tier + 1} ({o.upgrade_cost}g)"
        if kind == "refresh":
            return "roll"
        if kind == "toggle_freeze":
            return "freeze"
        if kind == "hero_power":
            return "hero_power"
        if kind == "dark_discovery":
            return "dark_discovery"
        if kind == "activate":
            return f"activate {o.board[args[0]].name}" if args[0] is not None else "activate"
    except Exception as e:  # keep playing whatever happens in logging
        return f"{kind}?"
    return kind


def _apply(game, p, st, idx, note, kind=None):
    if kind is None:
        a = action_from_index(idx)
        if a.kind not in ("end_turn", "reposition"):
            LOG.setdefault(id(game), {}).setdefault(p, []).append(_describe(game, p, idx))
    return _orig_apply(game, p, st, idx, note, kind)


play._apply = _apply


def on_turn_end(g, p, game, obs):
    rec = GAMES.setdefault(id(game), {"tribes": list(game.tribes), "players": {}})
    pl = rec["players"].setdefault(p, {"turns": []})
    if "hero" not in pl:
        pl["hero"] = HEROES.get(obs.hero_id, str(obs.hero_id))
    acts = LOG.get(id(game), {}).pop(p, [])
    pl["turns"].append(
        {
            "turn": game.turn,
            "tier": obs.tavern_tier,
            "gold_left": obs.gold,
            "max_gold": obs.max_gold,
            "health": obs.health + obs.armor,
            "board": [unit_dict(u) for u in obs.board],
            "hand": [u.name + ("*" if u.is_golden else "") for u in obs.hand],
            "trinkets": [TRINKETS.get(t, str(t)) for t in obs.trinkets],
            "frozen": bool(obs.is_frozen),
            "dd_left": obs.dark_discoveries_left,
            "actions": acts,
        }
    )


def main():
    mode, seed0, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    search_kind = "next_start_self"
    if "--search" in sys.argv:
        search_kind = sys.argv[sys.argv.index("--search") + 1]
    os.makedirs(out, exist_ok=True)
    model, ck = checkpoint.load(NET)
    opts = checkpoint_options(NET)
    opts.keep_replays = True
    opts.turn_stats = True
    models = {"agent": model, "rival": model}
    if search_kind == "end":
        spec = SearchSpec()
    elif search_kind == "next_start_self":
        spec = SearchSpec(horizon="next_start", opponents=ProbeSpec(model=NET, seed=32_500_000))
    else:
        spec = None
    if mode in ("allsearch", "mixed", "slow") and spec is not None:
        opts.search = spec
        opts.search_controllers = ("agent",)
    cfg = LobbyConfig()
    for i in range(n):
        seed = seed0 + i
        path = os.path.join(out, f"{mode}_{seed}.json")
        if os.path.exists(path):
            continue
        rng = random.Random(seed)
        if mode == "allsearch":
            ctrl = ["agent"] * 8
        elif mode == "mixed":
            ctrl = ["agent"] * 2 + ["rival"] * 6
        elif mode == "slow":
            ctrl = ["agent"] * 2 + ["sovereign", "9kgamerbot", "9kgamerbot", "9kgamerbot-tempo", "9kgamerbot-tempo", "heuristic"]
        elif mode == "selfplain":
            ctrl = ["rival"] * 8
        elif mode == "bots":
            ctrl = ["9kgamerbot"] * 4 + ["9kgamerbot-tempo"] * 4
        else:
            raise SystemExit(mode)
        rng.shuffle(ctrl)
        t0 = time.time()
        GAMES.clear()
        LOG.clear()
        res = play.play_games(cfg, [GameSpec(seed=seed, controllers=ctrl)], models, opts, record=(), on_turn_end=on_turn_end)[0]
        (rec,) = GAMES.values()
        rec.update(
            {
                "seed": seed,
                "mode": mode,
                "search": search_kind if opts.search is not None else "none",
                "controllers": ctrl,
                "placements": res.placements,
                "turns": res.turns,
                "net": NET,
                "seconds": round(time.time() - t0, 1),
            }
        )
        rec["players"] = {str(k): v for k, v in rec["players"].items()}
        with open(path, "w") as f:
            json.dump(rec, f)
        res.save_replay(os.path.join(out, f"{mode}_{seed}.replay.pb"))
        print(f"{mode} seed {seed}: {res.turns} turns, places {res.placements}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
