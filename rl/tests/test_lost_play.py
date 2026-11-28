"""A play that a crash ends (`play.LostPlay`: the engine panicked, or a worker died), and how
`train_lobby` goes on: a lost iteration is played again on other seeds, a lost check is skipped."""

from __future__ import annotations

import json
import os

import pytest

from seaglass_rl import play, train_lobby
from seaglass_rl.model import BattlegroundsMLP
from seaglass_rl.play import LEARNER, GameSpec, LobbyConfig, LostPlay, Player, PlayOptions


class PanicException(BaseException):
    """Stands in for `pyo3_runtime.PanicException`, which is not an `Exception` either."""


def _play(player: Player):
    model = BattlegroundsMLP(seed=0, kind_prior=True)
    specs = [GameSpec(seed=11, controllers=[LEARNER] * 8), GameSpec(seed=12, controllers=[LEARNER] * 8)]
    return player.play(LobbyConfig(), specs, {LEARNER: (dict(model.config), model.state_dict())}, PlayOptions())


def test_an_engine_panic_is_a_lost_play(monkeypatch) -> None:
    def panics(*args, **kwargs):
        raise PanicException("index out of bounds: the len is 5 but the index is 5")

    monkeypatch.setattr(play, "play_games", panics)
    with pytest.raises(LostPlay, match=r"index out of bounds.*seeds 11, 12"):
        _play(Player(0))


def test_other_errors_of_a_play_stay_what_they_are(monkeypatch) -> None:
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(play, "play_games", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _play(Player(0))


class _Dies:
    """A job that ends the worker that takes it: it unpickles as `os._exit(1)`."""

    def __reduce__(self):
        return (os._exit, (1,))


def test_a_dead_worker_is_a_lost_play_and_the_player_plays_on() -> None:
    player = Player(1)
    try:
        before = {w.pid for w in player.pool._pool}
        with pytest.raises(LostPlay, match="died while playing"):
            player._map([(_Dies(),)])
        assert player.pool is not None and not before & {w.pid for w in player.pool._pool}
        assert sorted(r.seed for r in _play(player)) == [11, 12]
    finally:
        player.close()


ARGS = [
    "--iterations", "1", "--games", "2", "--workers", "0", "--device", "cpu", "--threads", "1", "--minibatch", "256",
    "--epochs", "1", "--no-reposition", "--max-steps-per-turn", "12", "--eval-games", "2", "--no-eval-probe",
    "--no-eval-yardstick",
]  # fmt: skip


def _flaky(monkeypatch, lost: set[int]) -> list[list[int]]:
    """`Player.play` that loses the calls numbered in `lost` (from 0, not counting the rule bots'
    reference play that the trainer starts with). Returns the seeds of each call."""
    real, calls = Player.play, []

    def flaky(self, config, specs, *args, **kwargs):
        if kwargs.get("record") == ("heuristic",):
            return real(self, config, specs, *args, **kwargs)
        calls.append([spec.seed for spec in specs])
        if len(calls) - 1 in lost:
            raise LostPlay("the engine panicked (a test)")
        return real(self, config, specs, *args, **kwargs)

    monkeypatch.setattr(Player, "play", flaky)
    return calls


def test_train_lobby_plays_a_lost_iteration_again_on_other_seeds(tmp_path, monkeypatch, capsys) -> None:
    calls = _flaky(monkeypatch, {0})
    run = tmp_path / "run"
    train_lobby.main(["--run-dir", str(run), "--seed", "3", *ARGS])
    assert calls[0] == [3 * 10_000_019 + 1000, 3 * 10_000_019 + 1001]
    assert calls[1] == [seed + 500 for seed in calls[0]]
    (row,) = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    assert row["iter"] == 1 and "eval" in row
    assert "lost play 1 (the run stops past 3): iteration 1: the engine panicked (a test)" in capsys.readouterr().out


def test_train_lobby_skips_a_lost_check(tmp_path, monkeypatch, capsys) -> None:
    calls = _flaky(monkeypatch, {1})  # the first play of the check
    run = tmp_path / "run"
    train_lobby.main(["--run-dir", str(run), *ARGS])
    assert len(calls) == 2
    (row,) = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
    assert row["iter"] == 1 and "eval" not in row and (run / "latest.pt").exists()
    assert not (run / "evals").exists()
    assert "lost play 1 (the run stops past 3): the check of iteration 1" in capsys.readouterr().out


def test_train_lobby_stops_past_max_lost_plays(tmp_path, monkeypatch) -> None:
    calls = _flaky(monkeypatch, {0, 1})
    with pytest.raises(SystemExit, match="2 plays were lost to crashes"):
        train_lobby.main(["--run-dir", str(tmp_path / "run"), "--max-lost-plays", "1", *ARGS])
    assert len(calls) == 2
    _flaky(monkeypatch, {0})
    with pytest.raises(SystemExit, match="1 plays were lost to crashes"):
        train_lobby.main(["--run-dir", str(tmp_path / "strict"), "--max-lost-plays", "0", *ARGS])
