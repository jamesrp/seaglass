"""The human yardstick (`yardstick.py`), the style stats of a turn (`play.TurnStat`) and the forced
leveling curve (`curriculum.py`, `PlayOptions.forced_curve`, `train_lobby --force-curve`)."""

from __future__ import annotations

import json

import pytest
import seaglass

from seaglass_rl import checkpoint, curriculum, ladder, train_lobby, yardstick
from seaglass_rl.actions import ACTION_TO_INDEX
from seaglass_rl.model import BattlegroundsMLP
from seaglass_rl.play import LEARNER, GameSpec, LobbyConfig, Player, PlayOptions, board_type, play_games
from seaglass_rl.yardstick import YardstickSpec

UPGRADE = ACTION_TO_INDEX[("upgrade_tavern", None, None)]


def _model() -> BattlegroundsMLP:
    return BattlegroundsMLP(seed=0, kind_prior=True)


# -- The yardstick --------------------------------------------------------------------------------


def test_every_human_board_builds_in_this_engine() -> None:
    spec = YardstickSpec()
    data = yardstick.load(spec.name)
    full = yardstick.boards(spec.name)
    assert len(full) == len(data["boards"]) >= 10
    for entry, board, row in zip(data["boards"], full, yardstick.summary(spec.name)):
        assert 1 <= len(board) <= 7 and len(board) == len(entry["minions"])
        assert row["stats"] == sum(u.attack + u.health for u in board)
        for minion, u in zip(entry["minions"], board):
            assert (u.attack, u.health) == (minion["attack"], minion["health"])
            assert bool(u.is_golden) == bool(minion.get("golden", False))
            if minion["card"] is not None:
                assert u.name.replace("Golden ", "") == minion["card"]
    tenth = yardstick.boards(spec.name, 0.1)
    assert all(u.attack >= 1 and u.health >= 1 for board in tenth for u in board)
    assert sum(u.attack for b in tenth for u in b) < 0.11 * sum(u.attack for b in full for u in b)


def test_yardstick_scores_by_scale_and_only_on_its_turns() -> None:
    spec = YardstickSpec(sims=2)
    weak = [seaglass.parse_unit("2/2 beast t1")]
    huge = [seaglass.parse_unit("9000000/9000000 beast t1") for _ in range(7)]
    assert yardstick.score(spec, weak, 1, spec.min_turn - 1, seed=1) is None
    assert yardstick.score(spec, weak, 1, spec.max_turn + 1, seed=1) is None
    low = yardstick.score(spec, weak, 1, 12, seed=1)
    high = yardstick.score(spec, huge, 6, 12, seed=1)
    assert len(low) == len(high) == len(spec.scales)
    assert all(0.0 <= x <= 0.05 for x in low)
    # Not 1.0: one human board wins through Venomous, whatever the stats.
    assert all(0.8 <= x <= 1.0 for x in high)
    assert yardstick.score(spec, weak, 1, 12, seed=1) == low


def test_board_type() -> None:
    mech, beast, wild = (seaglass.parse_unit(f"1/1 {t} t1") for t in ("mech", "beast", "all"))
    assert board_type([]) == ""
    assert board_type([mech] * 3 + [beast] * 3) == ""
    assert board_type([mech] * 4 + [beast] * 3) == "mech"
    assert board_type([mech] * 3 + [wild] + [beast] * 2) == "mech"
    assert board_type([wild] * 7) == ""


def test_rating_reports_the_yardstick_and_the_style() -> None:
    res = ladder.rate("9kgamerbot", games=3, probe=None, player=Player(0), yardstick=YardstickSpec(sims=1))
    table = res["by_turn"]["agent"]
    assert all("stats" in e and "high" in e for e in table.values())
    scored = {t: e for t, e in table.items() if "human" in e}
    assert scored and all(10 <= t <= 16 for t in scored)
    for e in scored.values():
        assert len(e["human"]) == 2 and all(0.0 <= x <= 1.0 for x in e["human"]) and e["human_n"] >= 1
    assert res["meta"]["yardstick"]["name"] == YardstickSpec().name and res["meta"]["forced_curve"] is None
    types = res["behavior"]["agent"]["final_types"]
    assert abs(sum(types.values()) - 1.0) < 1e-9
    report = ladder.format_report(res)
    assert "human" in report and "h@10%" in report and "last boards by minion type" in report
    assert "vs humans" in ladder.format_summary(res)
    summary = ladder.human_summary(res)
    assert summary["scales"] == [1.0, 0.1] and summary["by_turn"]
    loaded = json.loads(json.dumps(res))
    assert ladder.format_report(loaded) == report and ladder.human_summary(loaded)["gate"] == summary["gate"]
    off = ladder.rate("9kgamerbot", games=3, probe=None, player=Player(0), yardstick=None)
    assert off["lobbies"] == res["lobbies"] and ladder.human_summary(off) is None
    assert all("human" not in e for e in off["by_turn"]["agent"].values())


def test_old_results_still_format() -> None:
    res = ladder.rate("heuristic", games=2, probe=None, player=Player(0), yardstick=None)
    for e in res["by_turn"]["agent"].values():
        del e["stats"], e["high"]
        e["kinds"].pop("forced_upgrade")
    del res["behavior"]["agent"]["final_types"], res["meta"]["yardstick"], res["meta"]["forced_curve"]
    assert "avg place" in ladder.format_report(res) and "avg place" in ladder.format_summary(res)


# -- The forced curve -----------------------------------------------------------------------------


def test_curves_parse() -> None:
    assert curriculum.parse_curve(None) is None and curriculum.parse_curve("off") is None
    assert curriculum.parse_curve("standard") == curriculum.CURVES["standard"]
    assert curriculum.parse_curve("7:4, 2:2".replace(" ", "")) == ((2, 2), (7, 4))
    for bad in ("nope", "2:9", "0:2", "2-3"):
        with pytest.raises(ValueError):
            curriculum.parse_curve(bad)
    curve = curriculum.CURVES["standard"]
    assert [curriculum.target_tier(curve, t) for t in (1, 2, 4, 5, 7, 9, 11, 20)] == [1, 2, 2, 3, 4, 5, 6, 6]
    assert curriculum.label(curve) == "standard" and curriculum.label(((3, 2),)) == "3:2" and curriculum.label(None) == "off"


def _play(options: PlayOptions, controllers: list[str], unforced: tuple[int, ...] = (), seed: int = 11):
    m = _model()
    spec = GameSpec(seed=seed, controllers=controllers, unforced_units=unforced)
    (res,) = play_games(LobbyConfig(), [spec], {LEARNER: m, "old": m}, options, record=(LEARNER,))
    return res


def test_forced_seats_follow_the_curve_and_the_upgrades_are_not_recorded() -> None:
    curve = curriculum.CURVES["standard"]
    base = dict(turn_stats=True, max_steps_per_turn=12, allow_reposition=False)
    controllers = [LEARNER] * 3 + ["old"] * 2 + ["heuristic"] * 3
    res = _play(PlayOptions(forced_curve=curve, **base), controllers, unforced=(2,))
    forced_upgrades = 0
    for u, stats in enumerate(res.turn_stats):
        n = sum(s.actions.get("forced_upgrade", 0) for s in stats)
        if controllers[u] == "heuristic" or u == 2:
            assert n == 0  # a rule bot, and the unit left off the curve
            continue
        forced_upgrades += n
        # An upgrade always fits the Gold of the curve's turn, so a forced seat is never below it
        # for more than a turn.
        for prev, s in zip(stats, stats[1:]):
            assert s.tier >= curriculum.target_tier(curve, prev.turn)
        assert any(s.tier >= 4 for s in stats if s.turn >= 8) or stats[-1].turn < 8
    assert forced_upgrades >= 4
    # The trajectories hold the policy's own decisions only: every forced upgrade is missing from them.
    own = sum(s.actions.get("upgrade_tavern", 0) for u in (0, 1, 2) for s in res.turn_stats[u])
    recorded = sum(int((t["actions"] == UPGRADE).sum()) for t in res.trajectories)
    assert recorded == own
    # With no curve, the same seed plays as before the option existed.
    plain = _play(PlayOptions(**base), controllers)
    assert all(s.actions.get("forced_upgrade", 0) == 0 for stats in plain.turn_stats for s in stats)


def test_forced_upgrades_are_marked_in_a_kept_replay(tmp_path) -> None:
    """A rating with `--replays` keeps the engine replay of a forced lobby: each forced upgrade is
    one step with the engine's `forced` note, and the replay still re-simulates."""
    curve = curriculum.CURVES["standard"]
    options = PlayOptions(forced_curve=curve, turn_stats=True, keep_replays=True, max_steps_per_turn=12, allow_reposition=False)
    controllers = [LEARNER] * 3 + ["old"] * 2 + ["heuristic"] * 3
    res = _play(options, controllers)
    counted = sum(s.actions.get("forced_upgrade", 0) for stats in res.turn_stats for s in stats)
    assert counted >= 4
    path = tmp_path / "school.replay.pb"
    res.save_replay(path)
    timeline = seaglass.Timeline.load(str(path))
    assert timeline.is_faithful
    # (The engine also marks the End Turn it makes for a seat at the step cap.)
    made = [f for f in timeline if f.kind == "act" and f.note and f.note.get("forced")]
    forced = [f for f in made if f.action.kind == "upgrade_tavern"]
    assert len(forced) == counted
    assert all("logp" not in f.note for f in forced)
    # The seat's own upgrades are its decisions: none of them is marked.
    own = sum(s.actions.get("upgrade_tavern", 0) for stats in res.turn_stats for s in stats)
    upgrades = [f for f in timeline if f.kind == "act" and f.action.kind == "upgrade_tavern"]
    assert len(upgrades) == counted + own


def test_forced_controllers_limit_the_curve_to_one_side() -> None:
    curve = curriculum.CURVES["to4"]
    opts = PlayOptions(turn_stats=True, max_steps_per_turn=10, forced_curve=curve, forced_controllers=("old",))
    controllers = [LEARNER] * 4 + ["old"] * 4
    res = _play(opts, controllers)
    for u, stats in enumerate(res.turn_stats):
        n = sum(s.actions.get("forced_upgrade", 0) for s in stats)
        assert (n > 0) == (controllers[u] == "old")


def test_checkpoints_are_rated_on_their_curve_only_when_every_seat_was_forced(tmp_path) -> None:
    m = _model()
    cases = (
        ({"force_curve": "standard"}, curriculum.CURVES["standard"]),
        ({"force_curve": "standard", "force_share": 1.0}, curriculum.CURVES["standard"]),
        ({"force_curve": "standard", "force_share": 0.5}, None),
        ({"force_curve": None}, None),
        ({}, None),
    )
    for i, (args, want) in enumerate(cases):
        path = tmp_path / f"m{i}.pt"
        checkpoint.save(path, m, iteration=1, args=args)
        assert ladder.checkpoint_options(str(path)).forced_curve == want


def test_versus_forces_each_side_as_its_own_options_say() -> None:
    m = _model()
    agent = (dict(m.config), m.state_dict())
    curve = curriculum.CURVES["to4"]
    cap = dict(max_steps_per_turn=10)
    out = ladder.versus(
        agent,
        agent,
        games=1,
        workers=0,
        options_a=PlayOptions(forced_curve=curve, **cap),
        options_b=PlayOptions(**cap),
        yardstick=YardstickSpec(sims=1),
    )
    assert out["meta"]["forced_a"] == "to4" and out["meta"]["forced_b"] is None
    forced = {side: sum(e["kinds"]["forced_upgrade"] for e in out["by_turn"][side].values()) for side in ("a", "b")}
    assert forced["a"] > 0 and forced["b"] == 0
    text = ladder.format_versus(out)
    assert "forced curve: to4" in text and "tier" in text
    json.dumps(out)
    with pytest.raises(ValueError):
        ladder.versus(
            agent,
            agent,
            games=1,
            workers=0,
            options_a=PlayOptions(forced_curve=curve, **cap),
            options_b=PlayOptions(forced_curve=curriculum.CURVES["fast"], **cap),
        )


def test_make_specs_leaves_units_off_the_curve_by_share() -> None:
    import random

    config = LobbyConfig()
    args = (config, 200, 4, [], ["heuristic"], 1.0, random.Random(0))
    every = train_lobby.make_specs(*args, seed=5)
    assert all(s.unforced_units == () for s in every)
    half = train_lobby.make_specs(config, 200, 4, [], ["heuristic"], 1.0, random.Random(0), seed=5, force_share=0.5)
    # The seating does not depend on the share.
    assert [s.controllers for s in half] == [s.controllers for s in every]
    left_off = sum(len(s.unforced_units) for s in half)
    assert 600 < left_off < 1000
    none = train_lobby.make_specs(config, 20, 4, [], ["heuristic"], 1.0, random.Random(0), seed=5, force_share=0.0)
    assert all(s.unforced_units == tuple(range(8)) for s in none)


def test_train_lobby_curve_options_and_guards(tmp_path) -> None:
    def parsed(*extra: str):
        return train_lobby.parse_args(["--run-dir", "x", *extra])

    assert train_lobby.play_options(parsed(), aux=True).forced_curve is None
    on = parsed("--force-curve", "standard")
    assert train_lobby.play_options(on, aux=True).forced_curve == curriculum.CURVES["standard"]
    assert train_lobby.play_options(on).forced_curve == curriculum.CURVES["standard"]
    half = parsed("--force-curve", "standard", "--force-share", "0.5")
    assert train_lobby.play_options(half, aux=True).forced_curve == curriculum.CURVES["standard"]
    assert train_lobby.play_options(half).forced_curve is None  # the evaluation: it must level by itself
    for extra in (["--pure", "--force-curve", "standard"], ["--force-curve", "nope"], ["--force-curve", "to4", "--force-share", "2"]):
        with pytest.raises(SystemExit):
            train_lobby.main(["--run-dir", str(tmp_path / "run"), "--iterations", "1", "--workers", "0", "--device", "cpu", *extra])


def test_train_lobby_runs_an_endgame_school_iteration(tmp_path) -> None:
    run = tmp_path / "school"
    train_lobby.main(
        [
            "--run-dir", str(run), "--iterations", "1", "--games", "2", "--workers", "0", "--device", "cpu",
            "--threads", "1", "--minibatch", "256", "--epochs", "1", "--no-reposition", "--max-steps-per-turn", "12",
            "--force-curve", "standard", "--eval-games", "2", "--no-eval-probe",
        ]
    )  # fmt: skip
    (row,) = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    assert row["eval"]["tier"]["7"] == 4.0 and "stats" in row["eval"] and "final_types" in row["eval"]
    _, ckpt = checkpoint.load(run / "latest.pt")
    assert ckpt["args"]["force_curve"] == "standard" and ckpt["args"]["force_share"] == 1.0
    assert ladder.checkpoint_options(str(run / "latest.pt")).forced_curve == curriculum.CURVES["standard"]
