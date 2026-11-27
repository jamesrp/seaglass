# Neural Opponents: Near-Term Sprint

*Written 2026-11-21. Measurements were taken at commit `711794e` in a 4-core Linux container with CPU PyTorch, on the default Solo lobby.*

This is the working plan for making the neural opponents stronger.

- [Section 1](#1-aims-and-decisions) states the aims and the decisions that scope the sprint.
- [Section 2](#2-the-sprint) is the sprint: five stages, each with a gate, and a sixth added by the [review after Stage 4](review-2026-11-27.md).
- [Section 3](#3-why-this-order-the-critique) is the critique of the current approach that the plan answers.
- [Section 4](#4-measurements) holds the measurements, and [Section 5](#5-caveats) says how far to trust them.

## 1. Aims and Decisions

The project goals are in the [top-level README](../README.md#project-goals). These decisions scope the sprint.

| Question | Decision |
| :--- | :--- |
| What comes first? | Strength. Difficulty levels and play-style variety are not goals. Speed at play time is welcome later and is not required now. |
| What is the objective? | Linear in placement: one point per opponent beaten, 7 for 1st down to 0 for 8th. |
| Which mode? | Solos on one fixed patch. Duos and patch updates come later. |
| What is the target design? | AlphaZero-style: learned policy and value, self-play, and a search designed by researchers. No hand-written play rules and no hand-engineered features at the level of single cards or situations. |
| What about the hand-written bots? | They keep their own utilities and stay as they are. They are opponents to bootstrap from and fixed points to measure against. |
| What hardware? | An Apple-silicon Mac for now. The CUDA path must keep working for other machines. |
| Is `bots/` protected? | No. It can change on this fork, and anything needed later can be recovered from git history. |

The existing PPO reward already matches the objective. `play.placement_reward` gives +1 for 1st down to -1 for 8th, which is `2 * points / 7 - 1`.

Reports use average place, where 4.5 is level and lower is better. Average points are 8 minus the average place.

## 2. The Sprint

Each stage has a gate. A later stage should not start until the gate before it passes, except Stage 3, which runs alongside Stages 1 and 2.

| Stage | Deliverable | Gate |
| :--- | :--- | :--- |
| 1. Measure | One command that rates any agent on a fixed ladder | Two runs of the same agent agree within the stated interval |
| 2. Fix the learner, then scale | New PPO defaults and one run of 100,000 games or more | A pure network beats 9kgamerbot at milliseconds per decision |
| 3. Raise throughput (parallel) | Observation encoding out of Python, batched inference | 5 times more games per core, with 10 times as the target |
| 4. Search inside the turn | Policy-guided search scored by the learned critic | Search plus net beats the net alone, then the distilled net beats the old net |
| 5. Freeze `bots/` | A pinned reference opponent, useful ideas ported to `rl/` | No hand-written card knowledge in the training path |
| 6. Teach the late game | A human yardstick on the ladder, then a network trained in lobbies that reach the late game | With no forced leveling, the network levels and wins more against human winner boards |

### Stage 1: Build the Measuring Instrument

Every later choice depends on telling a real gain from noise, and the current evaluation cannot do that. The trainer checks itself on 96 games, which is good to about ±0.45 places at two sigma.

Build one command that rates any agent, and use it for every claim. `seaglass_rl.evaluate` and `play.Player` already do most of the work.

- **Fast ladder.** Seat the agent in 2 of 8 seats, with two each of `9kgamerbot`, `9kgamerbot-tempo` and `heuristic`. Play 500 lobbies on fixed seeds with shuffled seats. That is 1,000 seat-games and about ±0.15 places at two sigma.
- **Slow ladder.** Add one `sovereign` seat and play about 100 lobbies. SovereignBot costs 30 to 50 seconds of CPU per game, so run this rarely.
- **Pin the anchors.** The ladder bots must not change between runs. Record the commit with every result.
- **Report** average place with its standard error, top-4 rate and first-place rate. Also report behaviour by turn: Tavern tier, Gold left unspent and actions taken.
- **Add a per-turn probe.** Fight each end-of-turn board against a frozen library of reference boards for the same turn, and report the win rate by turn. The combat simulator runs about 40,000 fights per second on one core, so this is nearly free.
- **Raise the trainer's own check.** `train_lobby --eval-games` defaults to 96 against seven HeuristicBots. Point it at the fast ladder instead.

The probe matters because placement arrives once per game and is noisy. Board strength by turn is a far denser signal, and it shows directly whether an agent is trading early strength for late strength.

**Gate.** Two runs of the same agent on different seeds agree within the stated interval. The rule-bot order from [Section 4](#4-measurements) reproduces.

#### Stage 1 Result (2026-11-21): the gate passes

The instrument is `python -m seaglass_rl.ladder` (`rl/seaglass_rl/ladder.py`, `probe.py`), described in the [`rl` README](../rl/README.md#rating-agents-ladderpy). `train_lobby` now rates itself on the fast ladder. The result files are in [`results/stage1/`](results/stage1/), taken at commit `2004905`, engine build `34fd1ee4a7df`, on the Mac mini's CPU with 2 to 3 worker processes. A fast-ladder rating took 8 seconds for a rule bot and 24 seconds for the MLP.

Two seeds of the same agent agree. The ± figures are one standard error, clustered by lobby. The difference column is at two sigma.

| Agent on the fast ladder | Seed 5,000,000 | Seed 5,500,000 | Difference |
| :--- | :--- | :--- | :--- |
| HeuristicBot | 5.06 ± 0.06 | 5.08 ± 0.06 | -0.03 ± 0.17 |
| 9kgamerbot | 4.32 ± 0.07 | 4.34 ± 0.08 | -0.02 ± 0.20 |
| Clone of HeuristicBot (60 games, MLP) | 5.88 ± 0.05 | 5.88 ± 0.06 | -0.00 ± 0.15 |

The rule-bot order reproduces. Rating `random` seats two each of four bots, the composition of the 400-lobby ranking in [Section 4](#how-the-rule-bots-rank). The anchors in that run came out at 3.04, 3.20 and 4.28 for 9kgamerbot, 9kgamerbot-tempo and HeuristicBot, against 3.13, 3.13 and 4.29 there. Random placed 7.48, against 7.46.

| Agent on the fast ladder | Average place | Top 4 | First |
| :--- | :--- | :--- | :--- |
| 9kgamerbot-tempo | 4.14 ± 0.07 | 56% | 14% |
| 9kgamerbot | 4.32 ± 0.07 | 54% | 16% |
| HeuristicBot | 5.06 ± 0.06 | 40% | 3% |
| Clone of HeuristicBot | 5.88 ± 0.05 | 23% | 3% |
| Random | 7.48 ± 0.01 | 0% | 0% |

The slow ladder took 12 minutes for 100 lobbies on 3 workers, about 21 seconds of CPU per lobby. Its file is marked dirty only because this document was being edited during the run. SovereignBot placed 1.27 ± 0.07 with 81% firsts, against 1.40 in the 48-lobby table of [Section 4](#how-the-search-bots-rank). 9kgamerbot as the agent placed 5.21 ± 0.15, and the anchors 4.71 to 5.31. So the slow ladder spends most of its resolution on SovereignBot's dominance, and it will only matter once an agent is near the top of the fast ladder.

Pairing two agents on the same seeds barely narrows the interval. 9kgamerbot against HeuristicBot is -0.73 ± 0.17 at two sigma paired and ± 0.18 unpaired. Once the agent's own play differs, a shared seed shares little luck, so the 500-lobby size is what buys the precision.

The probe shows each bot's curve, and it reproduces to within 0.02 across seeds. It is the win rate of the agent's end-of-turn board against 32 reference boards from the same turn.

| Probe by turn | 2 | 4 | 6 | 8 | 10 | 12 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| HeuristicBot | 0.46 | 0.63 | 0.59 | 0.46 | 0.34 | 0.32 |
| 9kgamerbot | 0.45 | 0.19 | 0.31 | 0.65 | 0.59 | 0.59 |
| 9kgamerbot-tempo | 0.45 | 0.51 | 0.55 | 0.54 | 0.47 | 0.49 |
| Clone of HeuristicBot | 0.37 | 0.45 | 0.41 | 0.32 | 0.29 | 0.34 |

HeuristicBot is strongest on turns 4 to 6 and fades. 9kgamerbot gives up board strength on turns 4 to 6 and is the strongest from turn 8. That is the tempo-against-greed trade the probe was meant to show. The clone is weaker than its teacher on every turn, which fits its deficit of 0.8 places.

### Stage 2: Fix the Learner's Defaults, Then Run It at Scale

The PPO pipeline in `rl/` is the right base. Its defaults push it toward buying stats every turn, and its runs are far too short. Change the defaults in one pass, then spend the compute on a single long run.

| Change | Where | Why |
| :--- | :--- | :--- |
| Apply λ once per turn, and set the per-turn γ to 1 | `ppo.gae`, `train_lobby --gamma` | Today λ = 0.95 decays per action, so credit halves every one to three turns. A level-up on turn 5 gets almost no credit for a win on turn 12. `architect/train_fast_dist_ppo.py` has a turn-level version. |
| Replace the scalar critic with an 8-way placement distribution | `model.py`, `transformer.py`, `ppo.py` | It is a better-conditioned target, and the linear value is its expectation. The Architect trainers have one, but with first-place-heavy atoms. Use linear atoms here. |
| Turn off stat-sum shaping | `train_lobby --shaping` (default 0.3) | The potential is a log of summed Attack and Health. It is blind to Deathrattles, auras and scaling, and it rewards stats now. If shaping is still needed, use the Stage 1 probe as the potential. |
| Take positioning out of the policy | `train_lobby --no-reposition`, plus a solver in the environment | The simulator can score a board order exactly. `bots/seaglass_bots/sovereign_positioning.py` already does this. |
| Use a stronger warm start | `pretrain.py` | It clones HeuristicBot today. 9kgamerbot is a full place better and plays 13 games per second. SovereignBot is about four places better again but slow, so use it for a smaller dataset and to relabel the learner's own states. |
| Widen the league | `train_lobby --p-heuristic`, `--pool-size` | Keep both 9kgamerbot variants in the field permanently. Sample past snapshots from the whole history, not only the last ten. |
| Add an auxiliary head for future board strength | model and loss | It gives the network a reason to represent scaling before the placement signal can teach it. |
| Pick the model by a matched run | `train_lobby --model` | The entity transformer scores each action from the card it touches, so it should generalise better than the flat MLP. Test that on the ladder before making it the default. |

Then make one run of 100,000 to 1,000,000 games. At the speed measured here, 100,000 games take about ten hours on four cores, so the low end is possible today.

The warm start and the rule-bot opponents are scaffolding. Once this stage passes its gate, repeat the run with `train_lobby --pure` and no warm start. That run is the check that the method works with no hand-written play at all.

**Gate.** A pure network, with no search, beats 9kgamerbot on the fast ladder.

#### Stage 2 Progress (2026-11-21): the gate passes on short runs, and the long run is next

The changes are in. Each one is a flag, and `rl/README.md` lists the flags that restore the old behaviour.

| Change | Status |
| :--- | :--- |
| λ once per turn, γ = 1 | Done: `--lam-per turn` and `--gamma 1` are the defaults. |
| 8-way placement critic, linear atoms | Done for the MLP and the transformer: `--value-head placement`, the default. Old scalar checkpoints still load. |
| Shaping off | Done: `--shaping 0`. |
| Positioning out of the policy | `--no-reposition` plus `--auto-position`, a budgeted search with the combat simulator that names no cards. It measured no gain, so it is off by default (below). |
| Stronger warm start | Done: `pretrain --teacher 9kgamerbot` is the default, with the trainer's move-rule flags. |
| Wider league | Done: both 9kgamerbots stay in the field (`--bots`, `--p-bot 0.3`), and snapshots come from the whole history (`--pool-recent 4 --pool-history 4`). |
| Auxiliary head for future board strength | Built: `--aux-coef` predicts the probe score 0, 2 and 4 turns ahead. Off by default until an A/B run. |
| Model chosen by a matched run | Done on a short run: the entity transformer wins by 1.7 places (below). |

The result files are in [`results/stage2/`](results/stage2/). All runs are on the Mac mini's CPU with 3 workers. The ± figures are one standard error.

**The warm start.** A clone of 9kgamerbot from 150 lobbies places 5.07 ± 0.07 on the fast ladder. The HeuristicBot clone from Stage 1 places 5.88. Two settings mattered.

| Clone of 9kgamerbot, 150 lobbies | Held-out action accuracy | Fast ladder |
| :--- | :--- | :--- |
| 4 epochs, critic trained | 74% | 6.03 ± 0.06 |
| 12 epochs, critic trained | 78% | 5.22 ± 0.06 |
| 12 epochs, critic untrained and zeroed | 77% | 5.07 ± 0.07 |

The cloned critic memorises each game's outcome. Every decision of a game shares its final place, so the old split by decision hid this. On whole held-out games its place cross-entropy rose from 2.17 after one epoch to 7.1 after twelve, against 2.08 for a uniform guess. PPO from that clone started with a value loss of 5.0, and its ladder place got worse, to 5.63 at iteration 25, before it recovered. `pretrain` now holds out whole games and, by default, leaves the critic untrained with its last layer zeroed. Memory caps the clone at about 150 lobbies: 157,000 decisions took a peak of 8 GB. Stage 3's packed storage would lift that cap.

**Positioning by search.** The 12-epoch clone placed 5.22 without `--auto-position`, 5.21 with it, and 5.25 with a search 16 times larger: 8 opponent boards, 8 fights each and 48 orders a turn. The moves were applied, about one a turn, and changed the outcome of 475 of 500 lobbies, but not the average. At this level of play, board order is not where the places are.

**A short PPO run passes the gate.** From the 12-epoch clone, with the new defaults, `--no-reposition` and `--lr 1e-4`, 150 iterations of 64 lobbies (9,600 games) took 34 minutes. The trainer's own fast-ladder check:

| Iteration | 25 | 50 | 75 | 100 | 125 | 150 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Average place | 5.63 | 4.87 | 4.28 | 4.00 | 3.82 | 3.74 |

On a fresh seed the iteration-150 network places 3.82 ± 0.06. 9kgamerbot on the same seed places 4.34 ± 0.08. The paired difference is -0.52 ± 0.18 at two sigma, z = -5.7. The network plays without search, so this meets the gate as written. It was still improving by about 0.1 places every 25 iterations.

Its style is narrow. Its mean Tavern tier reaches 3.8 by turn 8 and stays there, and from turn 9 it rolls 4 to 7 times a turn. The anchors reach tier 5.5 to 6. Its probe is 0.60 to 0.74 from turn 6 to turn 11 and fades to about 0.5 after turn 12. That is a tempo strategy that out-tempos the rule bots. Whether it holds against stronger play is what the long run, the slow ladder and the `--pure` repeat should show.

Speed: an iteration took 7 seconds at the start and 13 seconds by iteration 150, as games got longer and snapshots took opponent seats. The update was 40% to 60% of it.

**The transformer is far stronger.** The matched run cloned 9kgamerbot into the default entity transformer (64 wide, 3 layers) the same way, then ran the same 150 iterations. Its clone already placed 4.82 ± 0.07, against 5.07 for the MLP clone. After PPO:

| Iteration | 25 | 50 | 75 | 100 | 125 | 150 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformer | 3.48 | 2.66 | 2.44 | 2.22 | 2.18 | 2.04 |
| MLP | 5.63 | 4.87 | 4.28 | 4.00 | 3.82 | 3.74 |

The MLP run started from the clone whose critic had memorised its games, so the first 25 iterations are not a fair match. The gap at iteration 150 is too large for that to explain.

On a fresh seed the iteration-150 transformer places 2.08 ± 0.04, with 94% top 4 and 40% firsts. On the slow ladder it places 2.88 ± 0.08, second to SovereignBot at 1.55 ± 0.11, and well clear of the rule bots at 5.46 to 6.03. With 9kgamerbot in the agent seats SovereignBot placed 1.27 and took 81% of firsts; against the network it took 69%.

Its play differs from the MLP's. It often skips the turn-2 level, buying a second minion instead, and its probe on turn 2 is 0.78. It reaches a mean Tavern tier of 4.0 by turn 9 and keeps levelling, to 5.9. Its probe stays between 0.68 and 0.86 from turn 6 on, so it is not trading late strength for early strength.

It costs more to train. On 3 CPU threads its update took 20 to 32 seconds an iteration against 4 to 7 for the MLP, and the 150 iterations took 79 minutes. At that rate 100,000 games take about 14 hours on the CPU. The GPU gave the 64-wide transformer no speed-up in an earlier test, so that needs measuring before the long run.

**The long run.** On 2026-11-22 the transformer run continued from iteration 150 on 6 CPU cores. With the learning rate at 1e-4 it peaked at iteration 300, at 1.91 on the trainer's check, then slipped to 2.12 at 350 and 2.17 at 400. Head to head, iteration 400 lost to iteration 300, 4.63 against 4.37. So the run restarted from the iteration-300 snapshot at 5e-5, annealed linearly to 0 over 300 more iterations. That part used the Rust encoder and packed trajectories: about 32 seconds an iteration, 4 of them playing. It took 2 hours 51 minutes.

| Iteration | 150 | 300 | 375 | 450 | 525 | 600 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fast ladder, trainer's check | 2.04 | 1.91 | 1.85 | 1.85 | 1.80 | 1.79 |

The final network has seen 38,400 games along its line (600 iterations of 64), not the 100,000 planned: the transformer's update is the bottleneck on the CPU.

| Iteration-600 network | Result |
| :--- | :--- |
| Fast ladder, fresh seed | 1.79 ± 0.03, 98% top 4, 46% firsts (iteration 150: 2.08) |
| Slow ladder | 2.11 ± 0.06, with SovereignBot at 2.29 ± 0.14 (iteration 150: 2.88, SovereignBot 1.55) |
| Head to head with iteration 300 | 4.05 ± 0.06 against 4.95 |
| Head to head with iteration 150 | 3.72 ± 0.06 against 5.28 |

The network is now level with SovereignBot on the slow ladder, within the interval and slightly ahead, with no search and no hand-written knowledge. SovereignBot costs about 20 seconds of CPU a game; the network takes milliseconds a decision.

Its style moved again. It levels slowly, with a mean tier of 3.0 at turn 8 and 4.1 at turn 12, and from turn 7 it rolls 2 to 4 times a turn at low tiers. Its probe is 0.87 to 0.94 from turn 8 on. The anchors reach tier 5.5 by turn 10.

Lesson for later runs: anneal the learning rate. At a constant 1e-4 the policy drifted after about 300 iterations.

**More games (2026-11-23).** The run continued from iteration 600 on 6 CPU cores.

- **The first try drifted.** The learning rate restarted at 5e-5, annealed again. Head to head against iteration 600, iteration 700 lost 4.61 to 4.39 and iteration 800 lost 4.63 to 4.37. The trainer's check slipped from 1.79 to about 1.9. So it was stopped after 200 iterations.
- **The second try held.** It restarted from iteration 600 at 2e-5, annealed to 0 over 770 iterations, with rule bots in 15% of opponent seats instead of 30%. The bots now place about 5.5, so many bot seats mostly teach the network to exploit weak play. Iteration 700 was level with 600 head to head (4.47 against 4.53), and the run took 5 hours 16 minutes.

| Iteration | 600 | 700 | 900 | 1100 | 1300 | 1370 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fast ladder, trainer's check | 1.79 | 1.79 | 1.76 | 1.75 | 1.72 | 1.71 |

The iteration-1370 network has seen 87,680 games along its line.

| Iteration-1370 network | Result |
| :--- | :--- |
| Fast ladder, fresh seed | 1.75 ± 0.03, 98% top 4, 47% firsts (iteration 600: 1.79) |
| Slow ladder | 2.21 ± 0.08, with SovereignBot at 2.21 ± 0.12 (iteration 600: 2.11 against 2.29) |
| Head to head with iteration 600 | 4.20 ± 0.06 against 4.80 |

Head to head it is clearly stronger. The two slow-ladder runs (100 lobbies each) agree within their intervals: the network and SovereignBot are level.

Lesson: once a run has annealed, continue it at a much lower learning rate, and check snapshots head to head every 100 iterations or so. The fast ladder is too close to its floor to show drift.

Still to do in this stage: the `--pure` repeat and the A/B test of the auxiliary head (built, off by default: `--aux-coef`).

### Stage 3: Raise Throughput (in Parallel)

The engine is fast and the Python around it is slow. Engine calls take about 3% of rollout time. Building the observation in Python takes 43% to 59%.

Device selection is already done. Since `f425ecd` the gradient update runs on MPS or CUDA through `rl/seaglass_rl/device.py`, and self-play stays on CPU workers. On MPS the update of a 224-wide, 6-layer transformer runs about twice as fast, and the small default models gain little.

What remains is the rollout path.

- **Move observation encoding and the legal-action mask out of Python.** `features.encode_observation` and `actions.action_mask` should write straight into NumPy buffers from Rust. Test the result for exact parity with the Python encoder.
- **Step many lobbies per call** and run the network once per batch of decisions.
- **Store observations packed.** The dense format costs 40 KB per decision and is over 97% zeros. The documented 400-game pretrain ran out of memory in a 15 GB container because of it.

The README's rule is that Rust never hardcodes a neural network feature. To keep that true for the engine, put the encoder in a separate crate or module that depends on the engine. Where it lives is an open design point for the maintainers.

The gate is 5 times because of where the time goes. Removing encoding, masking and packing from Python leaves about 30% of today's cost. Batched inference and multi-lobby stepping are what take it toward 10 times.

**Gate.** Network rollouts run at 5 times today's games per core, from a baseline of 3.6 games per second with the MLP.

#### Stage 3 Result (2026-11-22): done at about 2x

James called this stage done on 2026-11-22. The 5x gate is not met, for the reasons below. The forward pass is where later speed-ups should come from.

Three of the items are in.

- **The encoder and the mask are in Rust.** `seaglass.Encoder` lives in the Python-bindings layer (`engine/src/python/encode.rs`), not in the game logic. Python still owns the layout: `features.py` passes in the vocabularies, the action table and every offset, and the Rust side holds only the formulas. Its output is bit-identical to `features.encode_observation` and `actions.action_mask` over 17,815 checked decisions, Solo and Duos. One call takes 8 µs against 86 µs in Python. `PlayOptions(rust_encoder=False)` keeps the Python path, which stays the reference.
- **Trajectories are packed.** The slot features travel as their nonzeros (about 109 a decision), the mask as bits and the card ids as uint16. That is 1.5 KB a decision against 21.5 KB, and the encoder hands the nonzeros over directly. Cloning a bot on 40 games peaked at 0.9 GB instead of 2.7 GB, and a 16-lobby rollout process at 0.4 GB instead of 1.1 GB.
- **The baseline moved.** This Mac is about 2.5 times faster per core than the container the baseline came from, so the comparisons below are against the Python path on the same Mac. Each is one core with 8 recorded seats.

| Policy | Python encoder, dense | Rust encoder, packed | Speed-up |
| :--- | :--- | :--- | :--- |
| Transformer, 64 wide, 3 layers | 2.2 games/s | 3.8 to 4.2 games/s | about 1.8x |
| MLP, 256 wide | about 9 games/s (fresh weights) | 13.4 to 13.9 games/s (cloned weights) | about 2.5x |

The MLP figures come from different weights, because a fresh network loses fast and plays shorter games, so treat that row as approximate.

Batched multi-lobby stepping is not done, and it cannot reach 5x. The network's forward pass is now the largest cost: 43% of an MLP rollout and 42% of a transformer one. The batches are already about 100 decisions wide, because `play_games` runs every lobby in a worker in lockstep. What is left outside the network is about a third of the time.

Training is also not rollout-bound now. A transformer iteration of 64 lobbies spends about 6 seconds playing and 30 seconds in the update, and more CPU threads did not shorten the update. So the next speed-up for training is in the update: the GPU when it is free, larger minibatches, or fewer epochs. More rollout speed would not help much.

### Stage 4: Add Search Only Where the Simulator Is Exact

Search pays when the model of the world is exact. In this game that means combat and the rest of the current turn. The future shop is hidden and random, so deep multi-turn search mostly measures its own guesses.

The ladder shows what search inside the turn is worth. SovereignBot's search with a hand-written evaluator finished about four places ahead of the best rule bot. This stage keeps that strength while a learned critic replaces the hand-written evaluator.

- Ship the positioning solver as part of the environment, from Stage 2.
- Add policy-guided search within a turn. Score each line with the learned placement critic at the start of the next turn.
- Once the critic has seen at least 100,000 games, use the search as the teacher. The Gumbel-style targets in `architect/gumbel_az.py` are the right tool.
- Keep `belief_tavern` as the only way search sees the future. It is what keeps the search honest about hidden information.

This is the AlphaZero loop the repo is already reaching for. The difference is that the value function is learned from enough games first, and every step is judged on the ladder. Search comes after Stages 1 to 3 because it needs their critic and their speed.

**Gate.** Search plus net beats the net alone, outside the interval. Then the net distilled from that search beats the previous net.

#### Stage 4 Progress (2026-11-22): the first half of the gate passes

`rl/seaglass_rl/search.py` searches a shopping decision with Gumbel sequential halving at the root:
- **Candidates.** The top 8 actions by the policy's logits (Gumbel noise off for play) are the candidates.
- **Lines.** Each candidate is continued by the policy in belief Taverns (`LobbyGame.belief_tavern`: the player's own state with a belief pool, so the search never sees the true hidden pool) until the policy ends its turn. That is 16 lines a decision in all, with the same seeds across candidates.
- **Scoring.** A line is scored by the placement critic on its end-of-turn observation. There is no combat, no future turn and no card knowledge.
- **Targets.** It also returns the completed-Q policy target, for distillation later.

`ladder.versus` seats two agents in 4 seats each and measures them against each other.

| Head to head, 200 lobbies | Average place |
| :--- | :--- |
| Iteration-600 network with search | 4.12 ± 0.06 |
| The same network without search | 4.88 ± 0.06 |

Search adds 0.76 places against the same network, well outside the interval. It costs about 7 seconds of CPU per searching seat per game, against well under a second without it, so it is a teacher and an evaluator, not a default.

Not done yet:
- **Ratings with search.** The search has not been rated on the slow ladder or the fast ladder.
- **Its knobs.** The candidates, lines, temperature and the σ scale (`c_visit` 50) are untuned.
- **Distillation works, but does not pass yet.** See the first try below.

#### Stage 4, First Distillation (2026-11-23): no gain yet

`distill.py` is expert iteration done offline. It plays lobbies where some seats search, records what the search preferred, and fine-tunes the network toward it.

- **The run.** The teacher was the iteration-1370 network (87,680 games). It searched in 2 seats of each of 2,000 lobbies, with Gumbel noise at the root, against copies of itself without search. That gave 806,481 recorded decisions, 747,341 of them searched, and took 53 minutes on 8 CPU workers. Each student fine-tuned for 2 epochs, about 4 minutes, on three terms: cross-entropy to the search's improved policy, the placement critic, and a KL anchor (0.1) to the teacher.
- **The students barely moved.**

| Student | Held-out policy cross-entropy | Top action agrees with search | KL to teacher | Head to head with teacher, 400 lobbies | Fast ladder |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Teacher, for reference | 0.561 | 89.7% | 0 | 4.50 by definition | 1.75 |
| lr 3e-5 | 0.558 | 89.6% | 0.004 | 4.45 ± 0.05 | 1.73 |
| lr 1e-4 | 0.560 | 89.2% | 0.008 | 4.46 ± 0.04 | 1.74 |

Both students edge the teacher by about 0.05 places, inside the interval. So the second half of the gate has not passed.

What this says:
- **The network and the search mostly agree.** Its top action is the search's top action on 90% of searched decisions, so the targets carry about 10% of new information.
- **The disagreements didn't transfer.** On held-out games the students never matched the search's choice more often than the teacher did. A likely reason: with 16 lines a decision, the search's verdict on close calls is noisy. Its 0.76-place edge in play may come from many small improvements that are hard to learn one decision at a time from 750,000 examples. That is a guess, and the next tries test it.

Two follow-ups ran the same night. Neither moved the result.

| Student | Held-out agreement with search (teacher's) | Head to head with teacher, 400 lobbies |
| :--- | :--- | :--- |
| Harder training: lr 3e-4, 4 epochs, no anchor | 88.1% (89.7%) | 4.45 ± 0.05 |
| Harder training: lr 1e-3, 4 epochs, no anchor | 86.4% (89.7%) | 4.52 ± 0.04 |
| Cleaner targets: 64 lines a decision, 500 lobbies, lr 3e-5 | 89.7% (89.9%) | 4.51 ± 0.05 |
| Cleaner targets: 64 lines a decision, 500 lobbies, lr 1e-4 | 88.9% (89.9%) | 4.46 ± 0.05 |

- **Harder training overfits.** It moves the students further, with a KL to the teacher of up to 0.06. Their agreement with search on unseen games falls instead of rising, and the critic's held-out loss grows.
- **More lines change little.** With 64 lines, the network still agrees with the search on 90% of decisions, and the students learn no more from the rest.
- **So noise in the targets is probably not the main problem.** The more likely limit is the student. The search computes each turn's consequences exactly in a simulator. The network is a 0.18M-parameter transformer that has to guess them in one forward pass, and it may not have the capacity to represent what the search adds.

Next tries:
- **A bigger student.** For example 128 wide and 4 layers (about 1M parameters), cloned from the current network, then trained on the search's targets and PPO. This is where the GPU earns its keep: an earlier test found MPS 2x faster than the CPU for a 224-wide, 6-layer transformer.
- **Distil inside PPO.** Act with search in a fraction of the training seats, so PPO learns from the better games, and add the search policy as an extra loss on those decisions.
- **Keep the search at play time.** At about 7 seconds of CPU per seat per game (50 to 100 ms a decision), the network plus search is already affordable as an opponent. That is the AlphaZero design anyway: the network guides the search, and the search plays.

#### Stage 4, a Bigger Student (2026-11-24)

`distill train --student 128,4,4,256` trains a fresh entity transformer, 128 wide and 4 layers, with 0.70M parameters, 4 times the teacher's. It is trained on both overnight data sets: about 1M decisions from 2,500 lobbies. There are three terms: the search's targets, the teacher's own policy (the anchor at weight 1.0, here acting as distillation), and the final places. It ran 8 epochs of about 5 minutes each on 6 CPU threads.

| | Held-out agreement with search | KL to teacher | Head to head with teacher, 400 lobbies | Fast ladder |
| :--- | :--- | :--- | :--- | :--- |
| Teacher (0.18M) | 89.3% | 0 | 4.50 by definition | 1.75 |
| Student (0.70M), epoch 8 | 84.1% | 0.063 | 4.54 ± 0.05 | 1.75 |

- **Almost the teacher, from scratch.** The student plays almost as well as the teacher, inside the interval, after one pass of supervised training. But it learned neither the teacher's policy exactly nor the search's refinements: its agreement with search is lower than the teacher's.
- **Its critic memorised the games.** Held-out value cross-entropy was best at epoch 2 (1.62) and reached 2.79 by epoch 8. With only 2,500 lobbies' outcomes, a fresh critic learns which game it is in, as the cloned critics did in Stage 2. It doesn't matter for play without search, but it would mislead search and PPO.
- **So this doesn't settle the capacity question.** A fresh student is limited by data before capacity.

**With the critic distilled too** (`--value-target teacher`, 6 epochs), the student's critic now learns the teacher's place distribution instead of the outcomes. Its held-out loss fell every epoch, to 1.58. The student ends level with the teacher head to head, 4.50 against 4.50 over 400 lobbies, at 85.0% agreement with search and a KL to the teacher of 0.053. On the fast ladder it places 1.72, against the teacher's 1.75. That makes it a sound starting point: a network 4 times larger, as strong as the old one, with a healthy critic, after 30 minutes on the CPU.

Next:
- **The real test of capacity.** Continue the bigger student with PPO, matched against the small network's own continuation. That is a long run, and it is where the GPU would help.

#### Stage 4, the Capacity Test (2026-11-24)

The test was two matched PPO continuations from iteration 1370, run side by side with both updates on the GPU:
- **Big:** the distilled 0.70M student.
- **Small:** the 0.18M network it was distilled from.
- **Same for both:** 500 iterations (32,000 games each), lr 2e-5 annealed to 0, 15% rule-bot seats, the same league history and fresh optimizer state. Sharing the GPU, the big run took about 60 seconds an iteration and the small one about 40. Both finished in about 7 hours.

| | Big (0.70M) | Small (0.18M) |
| :--- | :--- | :--- |
| Head to head, the two finals, 800 lobbies | 4.55 ± 0.03 | 4.45 ± 0.03 |
| Final against its own start, 400 lobbies | 4.43 against 4.57 | 4.46 against 4.54 |
| Slow ladder (SovereignBot in the same lobbies) | 2.16 ± 0.06 (2.12) | 2.21 ± 0.08 (2.22) |
| Fast ladder, trainer's check at the end | 1.73 | 1.71 |

**The bigger network did not pull ahead.** The finals are level head to head, with the small one marginally ahead. Each improved a little on its own start, inside the interval. At this training budget, the network's size is not what limits strength, so a bigger student isn't the route to distilling the search either. The search's measured gain (0.76 places at play time) remains the clearest one.

Next:
- **Improve the search itself.** First tune its knobs with short head-to-head runs. Then score lines at the start of the next turn, with this turn's combat simulated against sampled opponent boards (see "Later: Search Beyond the Turn").

#### Stage 4, Tuning the Search and Looking Past the Turn (2026-11-26)

Each variant searched in 4 seats of 400 lobbies against the iteration-1870 network without search in the other 4, on the same seeds. The table gives the searching side's average place (the network alone is 4.5 minus the gap, about 4.8 to 4.9). The difference column is paired by lobby against the default, at two sigma. Lower is better.

| Variant | Searching side | Against the default |
| :--- | :--- | :--- |
| Default: 8 candidates, 16 lines, line temperature 1, `c_visit` 50 | 4.125 | — |
| 32 lines | 4.138 | +0.013 ± 0.110 |
| 4 candidates | 4.195 | +0.070 ± 0.111 |
| 16 candidates, 32 lines | 4.227 | +0.102 ± 0.110 |
| Smaller σ scale (`c_visit` 10) | 4.274 | +0.149 ± 0.111 |
| Larger σ scale (`c_scale` 2) | 4.085 | -0.040 ± 0.111 |
| Larger σ scale, line temperature 0.5 | 4.084 | -0.041 ± 0.111 |
| Line temperature 0.5 | 4.113 | -0.012 ± 0.111 |
| Greedy lines (temperature 0) | 4.109 | -0.016 ± 0.117 |
| Scored at the start of the next turn, rule-bot opponent boards | 4.072 | -0.052 ± 0.113 |
| Scored at the start of the next turn, the network's own boards | 4.059 | -0.066 ± 0.113 |
| Scored at the end of the next turn, the network's own boards | 4.159 | +0.034 ± 0.112 |

- **The default knobs are near the optimum.** Only the smaller σ scale is clearly worse. Every other change is within ±0.11, the most 400 lobbies can resolve.
- **Scoring at the start of the next turn helps a little.** The line ends its turn, fights a sampled opponent board, starts the next turn (with its Gold and a fresh shop) and is scored there. That gave the best two results, 4.059 with the network's own boards and 4.072 with the rule bots'. Both are 0.05 to 0.07 better than the default, but inside the interval.
- **Scoring at the end of the next turn does not.** It came out at 4.159. Playing a whole extra turn on guessed shops and guessed opponents adds noise faster than information, as the Stage 4 caution predicted.
- **The confirmation was stopped early.** It was an 800-lobby rerun on a fresh seed, default against next-turn-start scoring with greedy lines. The default half finished at 4.187 ± 0.030. The second half was stopped when the weekend's runs were wrapped up early.

### Later: Search Beyond the Turn (to Evaluate)

These came out of a design discussion with James on 2026-11-23. Each one is to be built behind a flag and judged head to head (`ladder.versus`) and on the slow ladder, like everything else.

**How the AI picks an action today.** At each decision of its turn (a buy, a roll, a level, a Discover pick, and so on):
1. The network reads the observation. That is its own Tavern, the leaderboard (each opponent's Health, Tier, streak and hero), the damage cap, and the board its next opponent had when it last fought them.
2. Without search, it samples the policy, or takes its top action.
3. With search, it takes the 8 likeliest actions and plays each one forward to the end of the turn in belief Taverns. A belief Tavern is a copy of its own state whose card pool leaves out the cards it can see, so refreshes and Discovers draw what they plausibly could, never the hidden truth.
4. The policy plays the rest of each line, and the critic scores the line's end-of-turn state, before combat. Sixteen lines are spread over the candidates by Gumbel sequential halving, with the same random seeds for every candidate.
5. The best candidate is played on the real game, the real randomness is revealed, and the next decision searches again from scratch.

There is no tree below the root, no combat in the search and no model of opponents' coming boards. All of that is left to the critic, which learned it from outcomes.

| Idea | What it would add | Cost and risk |
| :--- | :--- | :--- |
| More lines per decision (for example 100 seeds instead of 16) | Less noise in each candidate's score. | Cost grows with the lines, about 7 s of CPU per seat-game now. The follow-up actions must stay the policy's, or another planner that sees only revealed information. A line that picks its later actions knowing the seed would roll because it knows the roll pays (clairvoyance). |
| Score at the start of the next turn, not before combat | The damage this turn's board takes or deals, made explicit. The roadmap's Stage 4 text asked for this; the current search stops before combat. | Needs a sample of the next opponent's board (below). Adds combat fights, which are cheap at about 10 µs each. |
| Look ahead to the end of the next turn (turn + 1, maybe + 2) | Curve and gold math made explicit. An example is James's turn-2 case: level, or buy Prisonguard and pump Scarlet Survivor, then see what turn 3's shop and 5 Gold allow. | Each extra turn multiplies the cost and compounds three guesses: our policy's later play, the shop draws, and the opponents' boards. The Stage 4 caution applies: past the turn the search "mostly measures its own guesses". Test whether it beats the critic's own estimate, early game first, where curves matter most. |
| A prior over the coming opponents' boards | The board to fight in the two ideas above. It could start as an empirical library: end-of-turn boards from recent self-play, keyed by turn and Tavern tier, like the probe's library but from the network's own lobbies. Later it could be a learned generative model. | The library is cheap and honest, since it holds only boards the lobby could produce. A learned model is a project of its own. |
| Every opponent's last-seen board in the observation, not only the next opponent's | More of what a person tracks about the lobby, for the policy and the critic. | A feature change: more slots, a retrained model, a new feature-spec version. |

### Stage 5: Freeze `bots/`

`bots/` should become a pinned reference, not the place where new work lands. Its evaluators hard-code card names and trinket priors from samples as small as 12 games, so part of it expires with every patch.

- Pin SovereignBot as the ladder's top anchor and as a teacher.
- Port four ideas into `rl/` with tests: the placement critic, turn-level advantage estimation, auxiliary heads and Gumbel targets.
- Keep hand-written card knowledge out of the training path. The static card features already give the network what it needs.

**Gate.** Nothing on the training path names a card.

#### Stage 5 Result (2026-11-26): the gate passes; the pure network is strong but behind the warm-started one

- **The gate.** `rl/tests/test_stage5.py` checks the modules `train_lobby --pure` trains with: the trainer, play, encoding, the network, PPO and the search. No string in them names a card, hero, trinket or Dark Gift. `--pure` games hold no rule bots, and `--pure` refuses everything that would bring rule-bot play into training: `--init`, bot seats, `--auto-position`, `--aux-coef`.
- **The bots stay as anchors.** The bots stay as the ladder's anchors and as opponents, and every rating now records an `anchors_fingerprint` of their sources.
- **The four ideas are ported into `rl/`, with tests.** They are the placement critic, turn-level advantages, auxiliary heads and Gumbel targets.

**The pure repeat.** It is the default transformer, trained from scratch with no warm start and no rule bots in training, on the GPU. It was planned for 3,000 iterations and stopped early, at iteration 795 (about 51,000 games), when the weekend's runs were wrapped up. Its last snapshot is iteration 790.

| Pure network, iteration 790 | Result |
| :--- | :--- |
| Fast ladder, fresh seed | 2.02 ± 0.04: far ahead of every rule bot (9kgamerbot places 4.31) |
| Slow ladder | 2.66 ± 0.11, with SovereignBot at 2.03 ± 0.11 |
| Head to head with the warm-started iteration-1870 network, 400 lobbies | 5.08 against 3.92 |

So the method works with no hand-written play at all: from nothing it passed every rule bot. On this budget it is clearly behind the line that started from a 9kgamerbot clone and had about 120,000 games. Its trainer's check was still improving slowly when it stopped:

| Iteration | 50 | 100 | 200 | 300 | 450 | 600 | 750 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fast ladder, trainer's check | 5.88 | 3.55 | 2.80 | 2.26 | 2.12 | 2.10 | 2.03 |

### Stage 6: Teach the Late Game

*Added 2026-11-27, after the [review after Stage 4](review-2026-11-27.md).*

The review played the best network (iteration 1870) and compared it with strong human play on this patch. Three findings set this stage.

- **The network wins on Tavern Tiers 2 and 3.** On turn 11 its seats average Tier 3.1, where strong players are on Tier 6. It has one real comp, Mech magnetize, and almost never holds a Tier 5 or 6 engine. Search does not change either.
- **Its boards are far below human boards.** On turns 12 to 15 they win 1% to 5% of fights against 13 published human winner boards, which hold about 20 times the stats. The rule bots and SovereignBot are in the same range, so every ladder so far has ranked agents inside a low-power field.
- **It is a stable point, so more of the same will not move it.** Forcing the network onto a human leveling curve costs it 0.57 places against its unforced self. Leveling pays only for a player who can turn Tier 5 and 6 shops into a scaling board, and nobody in the training pool can.

So the late game is an exploration problem. The learner never pays for it, so it never sees it, so it never learns that it pays. This stage puts the learner there directly, measures it against humans, and then takes the scaffold away.

| Step | Deliverable | Gate |
| :--- | :--- | :--- |
| 6a. Human yardstick | Every ladder rating reports the win rate against human winner boards, with the style behind it | The number reproduces between runs, and every later step is judged by it |
| 6b. Endgame school | A network trained with every model seat forced onto the standard leveling curve | Its boards grow several-fold on the forced curve, and Tier 5 and 6 engines appear |
| 6c. Take the scaffold away | The same network trained with fewer and then no forced seats, with the school network kept in the pool | With no curve it still levels, holds its human win rate, and beats iteration 1870 head to head |
| 6d. Lift the caps | A higher action cap, and board order re-tested on late boards | No loss of strength per hour of training |

**6a. The human yardstick** (`yardstick.py`). Placement and the probe are relative to the field. The yardstick is the absolute mark.

- It fights the agent's end-of-turn boards on turns 10 to 16 against a fixed set of human winner boards, in the combat simulator, at the published stats and at a tenth of them. The second number moves while the first is still near zero.
- The boards are data (`rl/seaglass_rl/data/human_boards/`). The first set is the 13 example winner boards from hspulse.net's comp pages for this patch. Boards from real game logs should replace it: `replay/` already rebuilds a game from a `Power.log`, and the Season 14 logs have to be saved before Patch 37.2 on 2026-12-01.
- A rating also reports the board's total stats and its Tier 5 and 6 minions by turn, and the minion types of the agent's last boards.
- **The gate number** is the win rate on turn 13 at the published stats.

**6b. Endgame school** (`curriculum.py`, `train_lobby --force-curve standard`). Every model seat is put on the standard curve by the environment: Tier 2 on turn 2, 3 on 5, 4 on 7, 5 on 9 and 6 on 11.

- Whenever a forced seat is below the curve and the Tavern upgrade is legal, the environment takes the upgrade. It is not one of the policy's decisions and is not recorded, so PPO does not train on it.
- Every model seat pays the same cost, so nobody is punished for leveling, and the learner spends turns 9 and later in Tier 5 and 6 shops against boards built there.
- The curve is a hand-written play rule. It is scaffolding in the same sense as the rule-bot warm start: `--pure` refuses it, and step 6c removes it.
- The run continues from iteration 1870 with the capacity test's recipe, a higher learning rate (the old one had annealed to zero) and a higher action cap.
- **If the boards grow several-fold and engines appear,** exploration was the block. Go to 6c.
- **If nothing changes with the whole lobby on Tier 6,** the block is structural: the action cap, the card representation, or credit across five or more turns. Lift the cap further (6d), add the auxiliary head that predicts the board's strength a few turns ahead (`--aux-coef`), and repeat.

**6c. Take the scaffold away** (`--force-share`). A network that was always leveled by the environment has never had to choose to level.

- Train on with half the model seats forced, then none. An unforced seat that is below the curve can tell it is on its own, so the policy can learn what leveling is worth there.
- Keep the endgame-school network in the opponent pool at 15% to 25% of seats, on its curve, so the learner keeps meeting boards that scale.
- A checkpoint trained with some seats unforced is rated with no curve. That is the only rating that says whether a network levels by itself.
- **Gate.** With no curve: mean Tier 4.5 or more on turn 10, the human win rate of 6b held, and a head-to-head win over iteration 1870.
- **If it slides back to Tier 2 and still wins,** the engine's balance of damage and tempo is the next suspect. Compare lobby length and damage per turn with the human logs: the network's lobbies last 17.4 turns, and the published human winners finish on turns 12 to 15.

**6d. Lift the caps.** The checkpoints so far were trained with at most 30 decisions a turn, and that cap already binds in 5% to 8% of turns 8 to 12. The top human comps of this patch take many actions a turn. Board order is also outside the policy (`--no-reposition`), and the position solver was off because it measured no gain on Tier 2 stat piles. Raise the cap toward the engine's own turn timer and measure the cost, and re-test `--auto-position` on boards with Taunts, cleaves and Deathrattle engines.

**After this stage.** Return to search and distillation. A critic that has seen engines pay has something for the search to find and for distillation to pass on. Until then the search stays at its current default.

**Not worth more time before then:** more iterations of the current recipe, a bigger network, more search knobs, or a longer search horizon scored by the same critic.

**The patch.** The engine models Patch 37.0.2. Patch 37.2 (Season 15) goes live on 2026-12-01 and removes Trinkets and Dark Gifts. This stage finishes on 37.0.2: a method that finds engines on one patch should find them on the next, and the hours from scratch to the gate are the cost of a new patch.

#### Stage 6 Progress (2026-11-27): the yardstick and the school are built

**6a is built.** `ladder rate`, `ladder versus` and the trainer's check now report the human win rate, the board's stats and Tier 5 and 6 minions by turn, and the minion types of the last boards. The baselines on the fast ladder, 500 lobbies each:

| Agent | Place | Tier on turns 7 / 10 / 13 | Median board stats, turn 13 | Tier 5 and 6 minions, turn 13 | Win rate against human boards, turn 13 | Same, human stats at 10% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Iteration 1870 | 1.77 ± 0.05 | 2.6 / 2.9 / 4.0 | 1,478 | 1.4 | 1.6% | 21.8% |
| Iteration 1870, another seed | 1.75 ± 0.05 | 2.6 / 2.9 / 3.9 | 1,446 | 1.3 | 1.5% | 21.6% |
| Iteration 1870 on the standard curve | 1.99 ± 0.07 | 4.0 / 5.0 / 6.0 | 1,316 | 2.6 | 1.2% | 19.8% |
| 9kgamerbot | 4.31 ± 0.14 | 4.0 / 5.4 / 6.0 | 642 | 2.6 | 0.1% | 8.2% |

- **The gate number reproduces.** Two seeds give 1.6% and 1.5% at the published stats, and 21.8% and 21.6% at a tenth of them. The review measured 2.6% in lobbies of the network against itself, where boards grow a little larger than against the rule bots.
- **The starting line for 6b is the third row.** Put on the curve with no training, the network holds twice the Tier 5 and 6 minions and builds a slightly weaker board with them.
- **Its last boards are 41% Mechs, 22% Pirates and 18% Dragons.** Beasts, Murlocs, Quilboar, Elementals and Aberrations are each 1% or less. 9kgamerbot's are spread over all ten types.

**6b is built.** `train_lobby --force-curve standard` passes its tests (`rl/tests/test_endgame_school.py`), and six iterations on CPU from iteration 1870 ran clean, with every model seat on Tier 4 on turn 7 and Tier 6 on turn 11.

- **The first run** (`runs/school/a` on the Mac mini): `--init` from iteration 1870, `--force-curve standard`, `--max-steps-per-turn 80`, `--lr 1e-4` held constant, `--p-bot 0.15`, 64 lobbies an iteration, and a check every 25 iterations. An iteration takes about 14 seconds there, so the run is set to 2,400 iterations or 10 hours, whichever comes first. The capacity test fine-tuned at 2e-5 annealed to zero and held its style. This run has to change its style, which is why it started with a fresh optimizer at 1e-4.
- **What to read.** The win rate against human boards on turn 13, the median board stats, the Tier 5 and 6 minions on the board, and the minion types. Place on the fast ladder is expected to slip and is not the measure here.
- **Its first check found a bug.** With `--replays` the check keeps replays, and the forced upgrade passed a replay note the engine does not take. The run stopped at iteration 25. The forced upgrade and the automatic reorder now carry the engine's `forced` note (commit `332387d`), and the run continued from its saved iteration 25 with `--resume`.
- **Its second check found a combat that never ends.** A `Heroic Broodmother` with the `Invulnerability` Dark Gift takes no damage while attacking and gets its Divine Shield back on every attack. Against a human board with thousands of Health it can neither die nor win, and the engine had no limit on combat length, so the fight ran until the worker ran out of memory and the check waited for it forever. The engine now calls a combat a draw after 1,000 attack turns ([`engine/docs/combat.md`](../engine/docs/combat.md) §5.2). The same fight can come up between two late boards in a lobby, so this guards the training games too. The run continued from its saved iteration 50.
- **A worker that dies now stops the run.** `multiprocessing.Pool` waits forever for the task of a worker that died, which is how the endless combat became a silent hang. `Player.play` now raises instead.

**Run A went the wrong way, and its learning rate is why.** Its checks on the curve, 500 lobbies each:

| Iteration | Place | Win rate against human boards, turn 13 | Same, human stats at 10% | Median board stats, turn 13 | Tier 5 and 6 minions, turn 13 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Start (iteration 1870 on the curve) | 1.99 | 1.2% | 19.8% | 1,316 | 2.6 |
| 75 | 1.95 | 1.1% | 19.0% | 1,244 | 2.5 |
| 100 | 2.06 | 1.0% | 18.6% | 1,256 | 2.2 |
| 150 | 2.03 | 0.8% | 16.0% | 1,078 | 1.9 |
| 200 | 2.03 | 0.8% | 16.4% | 1,092 | 1.9 |
| 250 | 2.10 | 0.5% | 14.1% | 1,000 | 1.5 |

- **The policy re-heated.** Its entropy rose from 0.46 to 0.57 in 100 iterations. That is the entropy `runs/stage2-tf` held at a constant 1e-4, with a ladder place near 2.0. The later runs cooled the network at 5e-5 and 2e-5, annealed, to an entropy of 0.49 and a place of 1.72. A constant 1e-4 undoes that cooling, and whatever the school teaches cannot be read under it.
- **It did learn to level.** Rated with no curve, the iteration-250 network is on Tier 3.1, 3.9 and 5.7 on turns 7, 10 and 13, against 2.6, 2.9 and 4.0 for the starting network, and on Tier 6 on turn 15. Its boards are weaker: 929 median stats on turn 13 and 0.3% against human boards, for a place of 2.05 ± 0.07.
- **Run A was stopped at iteration 278.** Its files are kept in `runs/school/a`.

**Run B closed the cost of the curve and did not find engines.** `runs/school/b` started from iteration 1870 again with `--lr 5e-5 --anneal-lr` over 1,800 iterations and everything else as in run A. It ran all 1,800 iterations (115,200 lobbies) in 7.6 hours. Its checks on the curve, averaged over 300 iterations (12 checks of 500 lobbies):

| Iterations | Place | Win rate against human boards, turn 13 | Same, human stats at 10% | Median board stats, turn 13 | Tier 5 and 6 minions, turn 13 | Dragons among the last boards | Policy entropy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 to 300 | 1.97 | 1.2% | 19.7% | 1,238 | 2.2 | 25% | 0.52 |
| 301 to 600 | 2.03 | 1.2% | 19.1% | 1,181 | 2.1 | 27% | 0.53 |
| 601 to 900 | 2.02 | 1.3% | 19.6% | 1,206 | 2.1 | 27% | 0.54 |
| 901 to 1,200 | 1.90 | 1.4% | 20.3% | 1,227 | 2.3 | 30% | 0.53 |
| 1,201 to 1,500 | 1.84 | 1.5% | 21.6% | 1,304 | 2.3 | 29% | 0.54 |
| 1,501 to 1,800 | 1.79 | 1.6% | 22.4% | 1,355 | 2.4 | 30% | 0.55 |

Its last network on the fast ladder, 500 lobbies each. The school plays with a cap of 80 actions a turn, so the starting network was rated again with that cap:

| Network | Curve | Place | Tier on turns 7 / 10 / 13 | Median board stats, turn 13 | Tier 5 and 6 minions, turn 13 | Win rate against human boards, turn 13 | Same, human stats at 10% |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Iteration 1870 | standard | 1.90 ± 0.07 | 4.0 / 5.0 / 6.0 | 1,314 | 2.6 | 1.2% | 19.9% |
| Iteration 1870, another seed | standard | 1.98 ± 0.08 | 4.0 / 5.0 / 6.0 | 1,276 | 2.6 | 1.2% | 19.9% |
| Run B, iteration 1800 | standard | 1.77 ± 0.05 | 4.0 / 5.0 / 6.0 | 1,397 | 2.5 | 1.8% | 23.1% |
| Run B, iteration 1800, another seed | standard | 1.78 ± 0.06 | 4.0 / 5.0 / 6.0 | 1,294 | 2.5 | 1.6% | 22.2% |
| Iteration 1870 | none | 1.74 ± 0.05 | 2.6 / 2.9 / 4.0 | 1,464 | 1.4 | 1.7% | 21.7% |
| Run B, iteration 1800 | none | 1.78 ± 0.05 | 2.6 / 3.0 / 4.3 | 1,238 | 1.0 | 1.1% | 18.6% |

Head to head against iteration 1870 with no curve, four seats each, 400 lobbies (`ladder versus`):

| Four seats of | Their place | Iteration 1870's place | Gap to 4.5 (positive is worse) |
| :--- | :--- | :--- | :--- |
| Iteration 1870 on the curve | 5.01 | 3.99 | +0.51 ± 0.08 |
| Run B's last network on the curve | 4.57 | 4.43 | +0.07 ± 0.09 |
| Run B's last network with no curve | 4.63 | 4.37 | +0.13 ± 0.09 |

- **The curve no longer costs places.** Put on the curve with no training, the starting network lost half a place to itself. After the school it is level. On the fast ladder it places 1.77 on the curve, where the starting network places 1.90 to 1.98 on the curve and 1.74 off it.
- **Against human boards it gained half a point.** It wins 1.6% to 1.8% on turn 13, from 1.2% on the curve, and 22% to 23% against human boards at a tenth of their stats, from 19.9%. That is where the starting network stands off the curve (1.7% and 21.7%).
- **The 6b gate is not met.** The gate asked for boards that grow several-fold and for Tier 5 and 6 engines. The median board on turn 13 holds 1,294 to 1,397 stats against 1,276 to 1,314 before, and 2.5 Tier 5 and 6 minions against 2.6.
- **The gains came late and had not stopped.** The first 900 iterations, with the learning rate between 5e-5 and 2.5e-5, are level with the start. Each 300 iterations after that gained on every measure, and the last check is the best one.
- **It does not level by itself.** With no curve it is on Tier 2.6, 3.0 and 4.3 on turns 7, 10 and 13, as the starting network is, and its boards are weaker than the starting network's. It never chose an upgrade in training, so this is step 6c's question.
- **The trainer's last check and the cloud rating agree to the digit** (1.77, 1.8%, 23.1%, 1,397), so a Mac checkpoint can be rated in either place.

**What the networks buy on Tier 6** (`rl/scripts/review/shoptake.py`, 120 lobbies of eight seats on the standard curve, shops seen with 3 or more Gold):

| Network | Tier 5 minions bought, of those offered on Tavern Tier 6 | Tier 6 minions bought | Tier 5 and 6 cards bought at least 1 time in 10, of 85 | Volumizers bought |
| :--- | :--- | :--- | :--- | :--- |
| Iteration 1870 | 1.9% | 3.3% | 8 | 67% to 75% |
| Run A, iteration 250 | 0.6% | 2.2% | 4 | 36% to 42% |
| Run B, iteration 230 | 2.2% | 4.2% | 11 | 45% to 49% |
| Run B, iteration 1800 | 3.3% | 6.1% | 10 | 51% to 66% |

- **On Tier 6 the starting network passes on almost every high-tier minion.** It buys 8 of the 85 Tier 5 and 6 cards it is shown with any regularity (`Falling Sky Golem`, `Hopebringer`, `Charging Czarina`, `Crimson Vindicator`, `Resourceful Robot` and three more) and never buys most of the rest. It still buys two of every three Volumizers.
- **Run B is moving toward Dragons.** By iteration 230, `Crimson Vindicator` went from 36% to 59%, `Bronze Timewalker` from 1% to 16%, `Persistent Poet` from 4% to 18%, `Firescale Hoarder` from 0% to 12%, `Felfire Conjurer` from 10% to 19% and `Heroic Broodmother` from 3% to 11%. The Volumizers fell from about 70% to about 47%. Its checks on the curve are level with the starting line so far (1.1% to 1.4% against human boards), so the new buys have not yet made stronger boards.
- **Run A unlearned.** Its take rates fell across the board, which fits the re-heating.
- **Run B's last network buys more of the same cards.** `Crimson Vindicator` ends at 96%, `Hopebringer` at 88% (from 62%), `Falling Sky Golem` at 86% (from 71%), `Charging Czarina` at 74% (from 57%), `Eternal Summoner` at 30% (from 14%) and `Heroic Broodmother` at 26%. The cards it tried at iteration 230 are back under 6%: `Bronze Timewalker` 1%, `Persistent Poet` 5%, `Firescale Hoarder` 1%. It never buys 51 of the 85 Tier 5 and 6 cards, the same count as the starting network (41 at iteration 230).

The counts are in [`docs/results/stage6/`](results/stage6/) (`shoptake_*.json`).

**What wins a place in a school lobby** (`rl/scripts/review/whatwins.py`, 400 lobbies of eight seats of one network, every seat still alive on turn 11, by quarter of board stats from weakest to strongest):

| Network | Seats alive on turn 11, of 3,200 | Lobby length, turns | Median board stats by quarter | Final place by quarter | Rank correlation with place: board stats | Same: Tier 5 and 6 minions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Iteration 1870, no curve | 2,700 | 17.5 | 562 / 849 / 1,165 / 1,652 | 5.02 / 4.36 / 3.74 / 2.58 | -0.47 | -0.14 |
| Iteration 1870 on the curve | 2,420 | 15.9 | 452 / 667 / 899 / 1,278 | 4.53 / 4.08 / 3.40 / 2.40 | -0.45 | -0.16 |
| Run B, iteration 790, on the curve | 2,406 | 15.6 | 407 / 640 / 903 / 1,337 | 4.53 / 4.00 / 3.43 / 2.37 | -0.45 | -0.20 |
| Run B, iteration 1800, on the curve | 2,430 | 15.8 | 464 / 719 / 978 / 1,394 | 4.52 / 4.01 / 3.37 / 2.52 | -0.43 | -0.12 |

| Tier 5 and 6 minions on the board, turn 11 | 0 | 1 | 2 | 3 | 4 or more |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Final place, iteration 1870 on the curve | 4.11 | 3.83 | 3.63 | 3.47 | 3.04 |
| Final place, run B at iteration 790 | 4.08 | 3.79 | 3.36 | 3.16 | 2.96 |
| Median board stats, run B at iteration 790 | 607 | 732 | 822 | 891 | 933 |

- **The size of the board on turn 11 decides the place far more than what is on it.** The strongest quarter of boards finishes two places ahead of the weakest. Four high-tier minions against none is worth about one place, and those boards are also half again as large, so part of that place is stats.
- **The curve makes the boards smaller and the lobby shorter.** The Gold for the upgrades comes out of the board: every quarter holds about 20% fewer stats on turn 11, 780 seats are dead by then against 500, and the lobby ends on turn 15.9 against 17.5.
- **A school lobby ends about five turns after the curve reaches Tier 6.** In those turns the larger board wins now, and a board that would scale has no time to. The curve puts the network in front of the Tier 6 shop, but the reward in a lobby of tempo boards still goes to tempo.
- **High-tier minions paid a little more in the middle of run B, and not at its end.** Their rank correlation with place went from -0.16 to -0.20 at iteration 790 and is -0.12 for the last network. Its boards on turn 11 are 3% to 9% larger than the starting network's in every quarter.
- **Dragons place best with the smallest boards.** Dragon boards finish at 3.34 to 3.40 on a median of about 590 stats, and Mech boards at 3.50 to 3.53 on 900 to 970. Total stats under-read what Dragons do in combat. Run B plays more of them: 693 Dragon seats alive on turn 11 against 507.

The counts are in `whatwins_*.json` in the same folder. The no-curve row was played with the plain cap of 30 actions a turn, and the curve rows with the school's 80.

**What Stage 6b says.** Leveling was not what kept the network from the late game. On Tier 6 it plays the game it already knew, a little better, with ten high-tier cards.

- **The school taught play from Tier 6, and nothing wider.** The cost of the curve is gone, the take rates on Tier 6 nearly doubled, and the cards behind that are the ones the starting network already bought.
- **A lobby of these boards pays for the board that is larger now.** It ends about five turns after the curve reaches Tier 6. A card that pays later, or only next to two others, is worth less than a Volumizer in the turns that are left, so the cards tried in the warm first half of run B were dropped again.
- **The plan's next step for this case was the auxiliary head and a higher cap.** The measurements point at something narrower: 51 of 85 high-tier cards are never bought, and five of the ten minion types together make up under 1% of its last boards. The network has no games in which those cards could pay.
- **Three ways forward are open, and the choice is James's:** a second scaffold that makes each school seat play one minion type, so that it has to make that type's cards work; learning from human games, which needs Season 14 logs saved before 2026-12-01 and puts human data on the training path; or step 6c with the network as it is. James chose the first on 2026-11-27, with no human data.

**In a lobby of one minion type the network is strong in three types and lost in the rest** (`rl/scripts/review/bytype.py`, 400 lobbies of eight seats on the curve, each lobby's shop holding one type and the typeless minions; boards on turn 13):

| The lobby's type | Median board stats, iteration 1870 | Same, run B's last network | Win rate against human boards, iteration 1870 | Same, run B's last network | Boards of the lobby's type, run B's last network |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Mech | 2,859 | 3,225 | 4.5% | 6.3% | 100% |
| Dragon | 1,274 | 1,732 | 4.3% | 5.3% | 100% |
| Pirate | 1,661 | 2,268 | 1.7% | 3.6% | 91% |
| Demon | 1,472 | 1,533 | 1.2% | 1.4% | 75% |
| Elemental | 1,215 | 1,483 | 0.6% | 1.2% | 55% |
| Aberration | 1,207 | 1,281 | 0.2% | 0.6% | 52% |
| Murloc | 1,030 | 1,266 | 0.5% | 0.9% | 53% |
| Quilboar | 1,224 | 1,162 | 0.6% | 0.9% | 48% |
| Undead | 1,060 | 1,070 | 0.2% | 0.6% | 78% |
| Beast | 728 | 856 | 0.4% | 0.4% | 47% |
| Mean over the types | 1,373 | 1,588 | 1.4% | 2.1% | 70% |

- **The school's gain shows more clearly here.** With only Mechs in the shop, run B's last network wins 6.3% against human boards on turn 13, and 49% against them at a tenth of their stats.
- **In six types its boards are weak.** With only Beasts in the shop its median board holds 856 stats on turn 13, a quarter of its Mech board, and half its boards there are a typeless pile. The published human Beast board holds about 52,000.
- **A one-type lobby is a place where those cards can pay.** Every seat is equally lost in a Beast lobby, so the first seat to make Beasts work wins it. That is the incentive the five-type lobby does not give, because Mechs, Dragons or Pirates are almost always on offer there.
- **`train_lobby --school-tribes 1:0.5` plays half the training lobbies with one type** (the type school, `curriculum.py`). The check stays on the five-type ladder.

**In a one-type lobby the network does not buy the type** (`shoptake.py --tribes 1`, 150 lobbies on the curve, minions offered from turn 9 on in shops seen with 3 or more Gold). The seven weak types are Aberration, Beast, Demon, Elemental, Murloc, Quilboar and Undead:

| Network | Minions of the lobby's type bought, weak types | Same, Tier 5 and 6 only | Same, Beast lobbies, Tier 5 and 6 | Minions of the lobby's type bought, Mech, Dragon and Pirate | Shop minions seen per lobby | Tier 5 and 6 cards never bought |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Run B, iteration 790 | 0.7% | 1.5% | 0.1% | 5.4% | 832 | 44 of 80 |
| Run B, iteration 1800 | 1.6% | 3.7% | 0.5% | 4.9% | 695 | 43 of 78 |
| Run D (type school), iteration 650 | 0.4% | 0.9% | 0.1% | 3.6% | 965 | 63 of 82 |
| Run D (type school), iteration 1175 | 1.0% | 1.9% | 0.5% | 4.1% | 881 | 49 of 81 |

- **With only Beasts in the shop, run B's last network buys 6 of the 1,135 Tier 5 and 6 Beasts it is shown** and none of 3,429 lower Beasts. It rolls and buys Tavern spells for the board it has: from turn 9 on in Beast lobbies, 32% of its actions are rolls, 12% buy a spell and 1% buy a minion.
- **Its board is full by turn 8, so a new minion replaces a grown one.** A Tier 6 minion off the shelf holds fewer stats than a Tier 2 minion with four turns of buffs on it. It is worth the slot only as an engine, over turns the lobby does not last.
- **Run D, the first type school run, tries fewer of these cards, not more.** (`runs/school/d` on the Mac mini, from run B's last network, `--school-tribes 1:0.5`, otherwise as run B.) At iteration 650 its weak-type boards were 10% to 35% larger than run B's at iteration 790, and it got them by rolling more and buying less. Its checks on the five-type ladder dipped to 0.8% against human boards and came back along run B's shape, to 1.16% at iteration 1175, where run B was at 1.45%.
- **At two thirds of the run its one-type boards are 5% larger than run B's last network's, with the same share of typed boards.** The mean over the types is 1,669 stats on turn 13 against 1,588, and 70% of its boards are of the lobby's type, as before. It buys 1.0% of the lobby's minions in the weak types, where run B's last network buys 1.6%. A shop of one type alone does not make it buy the type.

**A one-type shop needs a buy limit to teach the type** (`train_lobby --school-buys own`, `PlayOptions.school_buys`). A model seat of a school lobby can buy only minions of the lobby's types, so no Tavern spell and no typeless minion. The limit masks Buy actions, so every decision is still the policy's own and is trained on as usual. Rule bots, the five-type lobbies and every check are not limited. Run B's last network under the limit, before any training with it (`bytype.py --tribes 1 --buys own`, the same 400 lobbies; `--buys minions` allows every minion and no spell; boards on turn 13):

| The lobby's type | Median board stats, no limit | Same, only minions | Same, only the lobby's type | Boards of the lobby's type, no limit | Same, only the lobby's type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Mech | 3,225 | 3,275 | 3,512 | 100% | 100% |
| Dragon | 1,732 | 1,548 | 1,485 | 100% | 100% |
| Pirate | 2,268 | 1,717 | 1,521 | 91% | 99% |
| Demon | 1,533 | 1,480 | 1,464 | 75% | 99% |
| Elemental | 1,483 | 1,323 | 1,157 | 55% | 89% |
| Aberration | 1,281 | 967 | 653 | 52% | 84% |
| Murloc | 1,266 | 953 | 668 | 53% | 85% |
| Quilboar | 1,162 | 1,012 | 1,218 | 48% | 94% |
| Undead | 1,070 | 889 | 899 | 78% | 98% |
| Beast | 856 | 585 | 357 | 47% | 88% |
| Mean over the types | 1,588 | 1,375 | 1,293 | 70% | 94% |

- **Mech boards do not need spells or typeless minions.** They are 9% larger under the limit. Demon and Quilboar boards are level, Dragon boards lose 14% and Pirate boards a third.
- **In the other types the board falls to what the type's cards are worth in this network's hands.** A Beast board holds 357 stats on turn 13, a tenth of a Mech board, and Aberration and Murloc boards about 650. The typeless pile and the spells were holding those boards up.
- **Under the limit it buys more of the lobby's minions, and mostly it rolls** (`shoptake.py --tribes 1 --buys own`, 150 lobbies, from turn 9 on). In the seven weak types it buys 5.9% of the Tier 5 and 6 minions of the lobby's type it is shown, against 3.7% with no limit, and 1.5% of the lower ones, against 0.8%. In Beast lobbies it buys 49 of 1,530 high-tier Beasts and none of 4,412 lower ones, 57% of its actions are rolls and 1% buy a minion. 36 of 80 Tier 5 and 6 cards are never bought, against 43 of 78.
- **That gap is what the school can teach.** Every model seat of a school lobby is limited alike, so nothing else on the shelf can stand in for the type's cards, and the seat that makes them work wins the lobby.
- **Run E is run D with the limit** (`runs/school/e` on the Mac mini, started 2026-11-27 at 12:05 PM Pacific: `--school-tribes 1:0.5 --school-buys own --seed 3`, from run B's last network with the pool seeded with it, otherwise as run D). Run D was stopped at iteration 1207 of 1,800 to make room for it. Its snapshots are kept, so it can be resumed as the control.

Results are in [`docs/results/stage6/`](results/stage6/). The baselines were written on the code of commit `ab5b139` with only documentation uncommitted, which the files record as a dirty tree.

### Not in This Sprint

- Duos.
- Keeping a trained network useful across patches.
- Play-time speed.
- Grading human replays.

Difficulty levels and play-style variety are not planned at all.

## 3. Why This Order: the Critique

The direction of the repo is right and the order of work was wrong. It is building toward an AlphaZero-style learner, which suits a game with an exact combat simulator. It added search and hand-written evaluators before it could measure strength or train at scale.

### What Kind of Game This Is

Six properties decide what a training method has to be good at.

- **It is mostly eight solo games that meet in combat.** A player's shop, Gold and board are their own. Opponents matter through pairings, a shared card pool and a few Hero Powers. What self-play mainly supplies is the field: how strong boards are on each turn.
- **The signal per game is weak.** A seat makes about 100 decisions and gets one place out of 8. Shop rolls, Discover offers, pairings and combat rolls all add luck. Training needs many games, and so does telling two agents apart.
- **The key trade-off pays off late.** Levelling the Tavern or buying a scaling engine costs strength now and returns it 3 to 6 turns later, if the player survives. Buying the most stats each turn is a strong local optimum, and it is where RL gets stuck.
- **The content is large and changes every few weeks.** The engine has 252 minions, 110 heroes, 236 trinkets and 47 Dark Gifts. Anything written per card expires with the patch.
- **State and actions are sets of cards.** The board, hand and shop are variable-size sets, and most actions point at a card. That fits an attention encoder with per-card action scoring better than a flat vector with 330 fixed logits.
- **The simulator is exact where search needs it.** Inside a turn, buying, playing, selling and positioning are deterministic. Combat against a known board runs at about 40,000 fights per second. The cross-turn question of tempo against greed needs a learned value.

### What the Repo Does Today

There is no earlier roadmap document, so this is read off the code and its history.

| Layer | What it is | Where |
| :--- | :--- | :--- |
| RL pipeline | Behaviour cloning of a simple rule bot, then PPO in full lobbies against its own snapshots and that bot. A flat MLP by default, an entity transformer as an option. | `rl/seaglass_rl`, about 4,600 lines |
| Search bots | Beam search over the bot's own turn in a fair copy of its Tavern, scored by a hand-written evaluator, combat rollouts and a small value network. Three generations: Beam, Apex, Sovereign. | `bots/seaglass_bots`, about 18,000 lines |
| Architect experiments | 14 bot variants and 11 training scripts: composition planners, cloning the search bots into the transformer, PPO against Sovereign seats, an 8-way placement critic, Gumbel search with self-play. | `bots/seaglass_bots/architect`, about 22,000 lines |

Three facts about the current state shape the rest.

- **The strongest bots are not neural.** SovereignBot averaged place 1.40 against the rule bots. It scores states with a hand-tuned evaluator that names specific cards, and its value network is a small correction weighted 0.2.
- **No trained policy ships with the repo.** Checkpoints are git-ignored, and the only committed weights are five small value networks.
- **The experiments' own notes report the failures.** Docstrings record that search "Goodharts" the value network and that self-play PPO settles into low-tier tempo play. They also record held-out placement cross-entropy of 1.6 to 1.98, against 2.08 for a uniform guess.

### What It Gets Right

- **The engine.** It is deterministic, exact and fast, with a scenario file per card and tools that replay real game logs beside the engine's version. A whole 8-player game costs about 15 ms of engine time.
- **The boundary between rules and ML.** Python never reimplements a rule. Bots see only what a player sees, through `context` and `belief_tavern`.
- **The entity transformer.** It has one token per card and scores each action from the cards it involves.
- **Static card features.** Engine tags plus a rules-text embedding let rare and new cards borrow from similar ones. This is the main defence against patch churn.
- **Careful MDP choices.** These include per-turn discounting, the prior over action kinds, a separate positioning step and a zero-sum placement reward.
- **The right ingredients, already prototyped.** The Architect experiments contain a placement critic, auxiliary targets, turn-level GAE, positioning by simulation and on-policy search targets.

### Where It Goes Wrong

Six problems, ranked by cost. The first two cause most of the others.

**1. Results are judged on too few games, against too weak a baseline.**

- One seat's place has a standard deviation near 2.3, so a 96-game average is good to about ±0.45 places.
- HeuristicBot is the weak rule bot. It averaged place 4.29 in mixed lobbies against 3.13 for 9kgamerbot.
- The arena script defaults to 6 games. Commit claims such as "beating all existing bots" have no script, seeds or intervals behind them. The ladder here bears that claim out, but nothing in the repo could have shown it.
- Hard-coded priors rest on tiny samples. `architect/hybrid.py` scores trinkets from records like "41.7% 1st across n=12".

**2. Training runs are 100 to 10,000 times too small.**

- Value networks train on 64 to 320 games by default. The Architect PPO and Gumbel trainers play 1,000 to 19,000 games in all.
- A game gives each seat one outcome. 320 games are 2,560 outcomes, spread over 110 heroes, 236 trinkets and 252 possible tribe sets.
- The reported failures follow from this: search exploits value errors, and held-out cross-entropy barely beats a uniform guess.
- The runs were small because the teachers are slow. The search bots are over 1,000 times slower than the rule bots.

**3. The Python wrapper wastes the engine, and a GPU alone does not fix that.**

- Python feature encoding takes 59% of rollout time with the MLP and 43% with the transformer. The engine takes 3%.
- That gives 3.6 games per second per core with the MLP. The engine could supply about 20 times more.
- The "GPU-vectorised" Architect trainer reports 1,500 to 3,000 games an hour. That is slower than one CPU core of the plain pipeline, because one Python process encodes every observation.

**4. The RL setup leans toward the tempo trap.**

- Credit assignment is short, because λ decays per action.
- Shaping pays for raw stats.
- The warm start clones a bot that buys the biggest stats and plays everything.
- The opponent pool is the last 10 snapshots plus 30% HeuristicBots. If the learner drifts to tempo, so does its whole field, and greed never pays.
- Fixes exist in `architect/` but never came back to the `rl/` trainer.
- Positioning is learned from the final place through 42 Reposition logits, although the simulator can score an ordering directly.

**5. `bots/` has become hard to build on.**

- It is about 40,000 lines in three copy-forward generations, and 22,000 of them landed in one commit.
- Hand-written knowledge sits at its core: a composition registry, hero and trinket tier lists, and card names inside evaluators.
- None of it can produce data at the scale a network needs.

It is worth keeping for two jobs: a strong reference opponent, and a slow but strong teacher.

**6. The objective was unsettled.**

`rl/` rewards place linearly. The search bots use first-place-heavy tables, for example 2.20 for 1st, 0.55 for 2nd and -1.00 for 8th. Section 1 now settles this for the neural opponent: linear.

## 4. Measurements

The ± figures are one standard error.

### Where Rollout Time Goes

Share of single-core rollout time, by policy network.

| Share of rollout time | MLP | Transformer |
| :--- | :--- | :--- |
| Building the observation in Python | 59% | 43% |
| Network forward pass | 17% | 40% |
| Packing the trajectory | 8% | 5% |
| Engine calls | 3.5% | 2.8% |

| Workload on one core | Games per second |
| :--- | :--- |
| 8 HeuristicBots, nothing recorded | 41 |
| 8 HeuristicBots, observations recorded | 4.8 |
| 8 seats of the MLP policy | 3.6 |
| 8 seats of the transformer policy | 2.5 |

A PPO iteration of 48 games on four cores spent 4.9 seconds playing and 12.8 seconds on the update. The update was about 72% of an iteration on CPU.

### How the Rule Bots Rank

400 mixed lobbies with two seats per bot, shuffled every game.

| Bot | Average place | Top 4 | First |
| :--- | :--- | :--- | :--- |
| 9kgamerbot | 3.13 ± 0.06 | 73% | 25% |
| 9kgamerbot-tempo | 3.13 ± 0.06 | 76% | 21% |
| HeuristicBot | 4.29 ± 0.05 | 50% | 4% |
| Random | 7.46 ± 0.02 | 0% | 0% |

### How the Search Bots Rank

48 lobbies with one SovereignBot, one ApexBot and two each of the three rule bots.

| Bot | Average place | Top 4 | First |
| :--- | :--- | :--- | :--- |
| SovereignBot | 1.40 ± 0.08 | 100% | 62% |
| ApexBot | 1.98 ± 0.16 | 98% | 35% |
| 9kgamerbot-tempo | 5.14 ± 0.18 | 44% | 1% |
| 9kgamerbot | 5.43 ± 0.20 | 33% | 0% |
| HeuristicBot | 5.75 ± 0.17 | 24% | 0% |

The two search bots together used about 75 seconds of CPU per game. A rule bot uses well under a second.

### What the Documented Pipeline Produces

Average place over 400 games, with the agent in one seat.

| Agent | Against 7 HeuristicBots | Against 7 9kgamerbots |
| :--- | :--- | :--- |
| HeuristicBot itself | 4.50 by symmetry | 5.48 ± 0.09 |
| Clone of HeuristicBot (120 games) | 5.51 ± 0.12 | 6.14 ± 0.08 |
| PPO from the clone, iteration 20 | 5.29 ± 0.11 | 5.81 ± 0.09 |

- The clone matches its teacher on 91.4% of held-out decisions and still finishes last in 118 of 400 games. Small errors compound over about 100 decisions per game.
- Twenty PPO iterations, about 960 games, recovered a quarter of a place. The policy still sat on Tavern tier 2 through turn 5 and left 3.5 Gold unspent per turn.
- The trainer's own 96-game check scored that snapshot at 4.98 ± 0.23. The 400-game run put it at 5.29, so the built-in check was off by more than the whole gain.

### How These Were Taken

The scripts were throwaway and are not checked in. Stage 1 replaces them, and its first job is to reproduce these tables.

- **Speeds.** One process with `torch.set_num_threads(1)`, `play.play_games` on a default `LobbyConfig()`, eight seats of one controller. Shares come from `cProfile`.
- **Ladders.** `play.Player.play` with `GameSpec(seed=base + g, controllers=shuffled)` and `record=()`. The rule-bot ladder used base seed 5,000,000 and the search-bot ladder 6,000,000. Seats were shuffled with `random.Random(base)`.
- **Agent results.** `evaluate.evaluate(agent, [opponent], LobbyConfig(), games=400, agent_units=1, seed=2_000_000)`.
- **Clone.** `python -m seaglass_rl.pretrain --out runs/bc-solo.pt --games 120 --workers 4 --threads 4`.
- **PPO.** `python -m seaglass_rl.train_lobby --run-dir runs/solo --init runs/bc-solo.pt --lr 1e-4 --workers 4 --threads 4 --games 48 --eval-every 20`, stopped after iteration 20.

## 5. Caveats

- Every timing comes from a four-core cloud container. The ratios should carry over to other machines, and the absolute numbers will not.
- The search bots were rated on 48 games. The four-place gap is safe at that sample size, and the exact figures are not.
- The clone used 120 games because the documented 400 ran out of memory. The PPO run stopped at iteration 20. Both show where the pipeline starts, not its ceiling.
- The checkpoints that the `architect/` scripts refer to are not in the repo, so the neural policies they describe could not be rated.
- `bots/` was read through its entry points, trainers and docstrings, not every line.
- The sprint is a recommendation from measurements and code reading. None of its stages has been implemented or tested yet.
