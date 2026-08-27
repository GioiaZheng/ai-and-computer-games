# Atari Boxing Double-Dueling DQN

PettingZoo Atari Boxing project with a reproducible Double-Dueling DQN learner,
snapshot-opponent curriculum, fixed-seed evaluation, optional W&B tracking, and
CUDA support. The same policy can collect experience from either player role,
and model selection averages held-out results across both roles.

The environment reproduces the instructor's tournament wrapper order exactly:
max over two observations, four-frame action skipping, clipped rewards,
grayscale conversion, 84x84 resize, four-frame stacking, and a two-channel
agent indicator. The resulting observation is `uint8 (84, 84, 6)`.

## Source Context

This project is part of the independent `ai-and-computer-games` course-work repository. The setup follows the instructor-provided course repository:

https://github.com/Hautamaki-lab/Summer-School-2026

It is not a fork of that repository.

Lecture notes, slides, notebooks, generated checkpoints, and local result files are intentionally excluded from version control.

## Project Layout

```text
README.md                     Project overview and run commands
requirements.txt              Python package list
examples/boxing_random.py     Random PettingZoo Boxing rollout
examples/breakout_random.py   Random Gymnasium Breakout rollout
examples/dqn_simple.py        Small Breakout DQN practice script
src/dqn_boxing.py             Boxing DQN training script
src/ppo_boxing.py             PPO curriculum trainer for frozen opponents
src/export_ppo_policy.py      Convert an SB3 policy to submission weights
src/ppo_agent_template.py     Teacher-compatible PPO inference template
src/evaluate_boxing.py        Fixed-seed random/snapshot evaluation
src/play_dqn_boxing.py        Render a saved Boxing DQN checkpoint
tests/test_dqn_boxing.py      CPU architecture and update smoke tests
```

The project directory is kept for project metadata. Practice scripts live in `examples/`,
and the Boxing training code lives in `src/`.

## Environment

The project was tested in WSL with a conda environment named `pettingzoo`.

```bash
cd projects/boxing-dqn
conda create --name pettingzoo python=3.12 pip -y
conda activate pettingzoo
pip install -r requirements.txt
```

For CUDA-enabled PyTorch on WSL:

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
```

Quick CUDA check:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Quick Checks

Random Boxing rollout:

```bash
python examples/boxing_random.py
```

Random Breakout rollout:

```bash
python examples/breakout_random.py
```

Day 4 CPU/GPU-independent unit smoke tests:

```bash
python -m unittest discover -s tests -v
```

Small Boxing training smoke test:

```bash
python src/dqn_boxing.py --episodes 1 --max-steps 100 --learning-starts 20 --batch-size 8 --replay-size 500 --target-update 50 --opponent random --eval-every 1 --eval-episodes 2 --save-every 1
```

Expected generated files:

```text
checkpoints/day4_boxing_latest.pt
checkpoints/day4_boxing_best.pt
results/day4_boxing_training.csv
results/day4_boxing_evaluation.csv
```

These files are ignored by git.

## Longer Run

```bash
python src/dqn_boxing.py \
  --episodes 200 \
  --device cuda \
  --opponent mixed \
  --train-role random \
  --replay-size 20000 \
  --learning-starts 2000 \
  --eval-every 10 \
  --eval-episodes 10 \
  --wandb-mode online
```

The mixed curriculum uses random opponents for initial data collection and then
mixes random play with a periodically frozen snapshot. One opponent is fixed
for each episode to reduce within-episode non-stationarity. With
`--train-role random`, one learner role is also fixed per episode, while the
shared Q-network learns from both `first_0` and `second_0`. Periodic evaluation
reports both roles separately and uses their combined mean for model selection.

Resume an interrupted run:

```bash
python src/dqn_boxing.py --episodes 200 --device cuda --resume checkpoints/day4_boxing_latest.pt
```

Evaluate the best checkpoint on 100 held-out seeds:

```bash
python src/evaluate_boxing.py --checkpoint checkpoints/day4_boxing_best.pt --eval-episodes 100 --device cuda
```

Watch the agent after training:

```bash
python src/play_dqn_boxing.py --checkpoint checkpoints/day4_boxing_best.pt --episodes 1
```

## Isolated PPO Challenger

PPO experiments use a separate environment so that adding SB3 does not modify
the tournament-tested `pettingzoo` environment:

```bash
conda create --name boxing-ppo --clone pettingzoo -y
conda activate boxing-ppo
pip install stable-baselines3==2.9.0
```

The PPO wrapper keeps the instructor's environment pipeline unchanged. It
randomizes the learner's side each episode and gradually replaces random
opponents with teacher-compatible frozen agents:

```bash
python src/ppo_boxing.py \
  --external-opponent /path/to/first_agent \
  --external-opponent /path/to/second_agent \
  --timesteps 300000 \
  --n-envs 4 \
  --device cuda \
  --output checkpoints/ppo_boxing
```

Exporting produces a plain PyTorch state dict. The submission itself does not
import SB3:

```bash
python src/export_ppo_policy.py \
  checkpoints/ppo_boxing.zip \
  checkpoints/ppo_policy_weights.pt
```

Resume with an additional number of timesteps:

```bash
python src/ppo_boxing.py \
  --resume checkpoints/ppo_pool_100000_steps.zip \
  --timesteps 200000 \
  --output checkpoints/ppo_pool_300000
```

For a random-opponent phase, CPU environment simulation can run in parallel:

```bash
python src/ppo_boxing.py \
  --resume checkpoints/ppo_pool_300000.zip \
  --external-probability 0 \
  --vec-env subproc \
  --n-envs 6 \
  --entropy-coefficient 0.002 \
  --timesteps 700000 \
  --output checkpoints/ppo_random_1m
```

External PyTorch opponents stay on `--vec-env dummy` so CUDA models are not
created inside environment subprocesses.

## Algorithm

The learner combines:

- 84x84 grayscale frame stacks;
- the instructor's two-channel player-role indicator;
- compact uint8 CPU replay;
- a convolutional dueling value/advantage head;
- Double-DQN targets;
- Huber loss and gradient clipping;
- a delayed target network;
- linearly annealed epsilon-greedy exploration;
- random and frozen snapshot opponents;
- random, first-only, or second-only learner-role sampling;
- periodic two-role fixed-seed evaluation and best-checkpoint selection.

Generated checkpoints, CSV results, W&B runs, course notes, and reports are
excluded from version control. Training source code and tests are tracked.
