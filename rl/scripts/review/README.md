# Review Scripts (2026-11-27)

These are the scripts behind [`docs/review-2026-11-27.md`](../../../docs/review-2026-11-27.md), kept as they were run. They are analysis tools, not part of the trainer.

Two of them became features and are better used there:

- The fights against human winner boards (`metaprobe.py`) are now part of every ladder rating: [`seaglass_rl/yardstick.py`](../../seaglass_rl/yardstick.py).
- The forced leveling curve (`curve_exp.py`) is now an environment option with tests: [`seaglass_rl/curriculum.py`](../../seaglass_rl/curriculum.py), `train_lobby --force-curve`, `ladder rate --force-curve`.

## Running Them

Run from `rl/` with the package on the path. `REVIEW_DIR` is the folder the logs go to and are read from (default `review`), and `NET` is the checkpoint that plays (default `runs/capacity/small/latest.pt`).

```bash
cd rl
export PYTHONPATH=. REVIEW_DIR=review NET=runs/capacity/small/latest.pt
python scripts/review/watch.py allsearch 9200000 10 $REVIEW_DIR/games    # ten lobbies, every seat searching
python scripts/review/watch.py selfplain 9300000 200 $REVIEW_DIR/games   # 200 lobbies without search
python scripts/review/analyze.py selfplain                               # leveling, rolls, final boards
python scripts/review/show.py $REVIEW_DIR/games/allsearch_9200000.json winner
python scripts/review/metaprobe.py selfplain --scale 0.1
python scripts/review/curve_exp.py standard 9400000 200 $REVIEW_DIR/games
python scripts/review/shoptake.py 9500000 120 $REVIEW_DIR/shoptake.json --curve standard --cap 80
python scripts/review/whatwins.py 9700000 400 $REVIEW_DIR/whatwins.json --curve standard --cap 80 --turn 11
```

`watch.py` writes one `MODE_SEED.json` (the per-turn log of all eight seats) and one `MODE_SEED.replay.pb` (the engine replay) per lobby. It runs one lobby at a time in one process, so start several with different seed ranges to use more cores.

| Script | What it does |
| :--- | :--- |
| `watch.py` | Plays lobbies (all-search, mixed, slow ladder, plain self-play, rule bots) and writes the per-turn log and the replay. |
| `analyze.py` | Tavern Tier, Gold, rolls, buys and board stats by turn; the first turn on each Tier; the most common cards on final boards. |
| `show.py` | Prints one seat's game turn by turn, or a lobby summary. |
| `origins.py` | Where the Tier 5 and 6 minions on a board came from; what a seat buys on each Tavern Tier; how often the engine cards appear. |
| `metaprobe.py` | Fights end-of-turn boards, read back from the replays, against the 13 human winner boards, with a `--scale` for the human stats and the win rate by comp. |
| `curve_exp.py` | The forced-leveling what-if: four seats forced onto a curve against four plain seats. |
| `temp_exp.py` | The greedy-play what-if: four seats at a given temperature against four sampling seats. |
| `shoptake.py` | What a net buys from the shops it is shown: minions offered and bought, by the seat's Tavern Tier and the minion's Tier, and by card. `--curve standard` plays it as endgame school does. |
| `whatwins.py` | What wins a place among the net's own seats: the average final place by the Tier 5 and 6 minions, the total stats and the minion type of a seat's board on one turn. |
| `export.py` | Collects the numbers behind the review's two charts into `chartdata.json`. |
| `cardtext.py` | Pulls each card's rules text from the engine's doc comments. |

## Caveats

- A replay re-simulates only on the engine commit that recorded it. The review's replays were recorded at `63a22e1`.
- `watch.py` patches `play._apply` to log every action by name, so it follows `play.py`'s internals and may need care after changes there.
