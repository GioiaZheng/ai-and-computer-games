# Combat Tank Agent

Training and evaluation code for the PettingZoo Atari Combat Tank environment,
developed for the final project of the University of Eastern Finland course
**Artificial Intelligence for Computer Games**.

> Course reference: the project follows the environment specification and
> practical context published in
> [Hautamaki-lab/Summer-School-2026](https://github.com/Hautamaki-lab/Summer-School-2026).
> This is an independent implementation, not a fork of the course repository.

The final submission is a role-specialized policy that controls either player,
navigates the maze, locates moving opponents, and maximizes the unchanged
official hit score. The evaluation-time implementation and selected checkpoint
are available in `submission/`.

## Environment

The project uses the instructor-provided scoring configuration without changing
its rules:

```python
from pettingzoo.atari import combat_tank_v2

combat_tank_v2.env(
    has_maze=True,
    is_invisible=False,
    billiard_hit=False,
)
```

This is the PettingZoo 1.26.1 module equivalent of the instructor's
`make("aec", "atari/combat_tank-v2", ...)` notation. Training uses
`combat_tank_v2.parallel_env(...)` with the same three game parameters.

Observations preserve RGB player identity, are resized to 84 x 84, and are
stacked over four frames inside the learning adapter. The PettingZoo environment
still advances exactly once for every selected action.

## Training Curriculum

The project separates navigation from combat instead of expecting sparse hit
rewards to teach every skill at once:

1. Behavioral cloning provides basic movement and firing initialization.
2. Count-based frontier exploration rewards newly reached maze cells and
   penalizes prolonged stagnation.
3. Tactical PPO fine-tuning learns target reacquisition and movement away from
   recently exposed firing positions.
4. Mixed-opponent training samples random, scripted, and frozen policies.
5. Checkpoints are selected using unshaped official scores from both roles.

The exploration reward is used only during training. Evaluation and tournament
selection use the official `+1` hit and `-1` received-hit rewards.

## Setup

```bash
conda activate boxing-ppo
pip install -r requirements.txt
```

## Reproduce the environment

```bash
python -m src.random_baseline --episodes 10 --render
```

## Train Frontier Exploration

```bash
bash scripts/train_maze_frontier.sh
```

The run preserves all 18 actions, balances the two roles, saves checkpoints
every 50,000 steps, and writes a local log under `results/logs/`.

## Train Combat Tactics

After selecting a navigation checkpoint:

```bash
bash scripts/train_combat_tactics.sh
```

This stage removes fixed learner and opponent prefixes. It trains against a
mixture of opponent behaviors and adds short-lived training signals for moving
away from a recently hit position. Internet access and W&B are not required.

## Behavioral Cloning

```bash
python -m src.pretrain_bc --episodes-per-role 4 --epochs 150 --device cuda
```

## Evaluate

```bash
python -m src.evaluate \
  --model checkpoints/candidate/model.zip \
  --opponent random \
  --action-set all \
  --games-per-role 10
```

Evaluation reports both roles separately, hit and received-hit counts, wins,
draws, combined official score, and action coverage. Maze coverage can be
measured independently:

```bash
python -m src.evaluate_navigation \
  --model checkpoints/candidate/model.zip \
  --games-per-role 3 \
  --deterministic
```

## Tournament Submission

The evaluator constructs `Agent(env)` and calls `get_action(state)`. The final
package follows that interface without changing the instructor-specified
environment, observation wrappers, rendering, or reward:

```text
submission/
|-- agent.py
`-- weights.pt
```

`agent.py` detects the controlled role from the SuperSuit agent-indicator
planes and selects the corresponding policy head. CUDA is used when available,
with CPU inference as a fallback.

## Verified Baselines

All values below use the official `billiard_hit=False` configuration and random
opponents. Earlier exploratory checkpoints used different rules and are not
used for model selection.

| Candidate | Games | `first_0` | `second_0` | Combined | Positive games |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform random | 16 | 0.00 | 0.00 | 0.00 | 0/16 |
| Scripted role-specific hunter | 40 | +3.55 | +3.15 | **+3.35** | 37/40 |
| Best 200k sweep PPO | 8 | +2.25 | +2.25 | +2.25 | 7/8 |

These early baselines are retained for context. New checkpoints are promoted
only after held-out evaluation against several opponent types and seeds.

## Repository Scope and Attribution

This public repository contains the team's implementation, tests,
reproducible experiment scripts, final report, and selected tournament agent.
Lecture PDFs, slides, instructor source files, third-party agents, intermediate
checkpoints, and generated logs are not redistributed.

References:

- [Hautamaki-lab/Summer-School-2026](https://github.com/Hautamaki-lab/Summer-School-2026)
- [PettingZoo Combat: Tank](https://pettingzoo.farama.org/environments/atari/combat_tank/)
- [Multi-Agent Reinforcement Learning: Foundations and Modern Approaches](https://www.marl-book.com/)

## Team

- Gioia Zheng
- ZeYang Fu
- Gankhulug Bayaraa
