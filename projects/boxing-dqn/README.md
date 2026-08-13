# Atari Boxing Double-Dueling DQN

PettingZoo Atari Boxing project with a reproducible Double-Dueling DQN learner,
snapshot-opponent curriculum, fixed-seed evaluation, optional W&B tracking, and
CUDA support.

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
  --replay-size 20000 \
  --learning-starts 2000 \
  --eval-every 10 \
  --eval-episodes 10 \
  --wandb-mode online
```

The mixed curriculum uses random opponents for initial data collection and then
mixes random play with a periodically frozen snapshot. One opponent is fixed
for each episode to reduce within-episode non-stationarity.

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

## Algorithm

The learner combines:

- 84x84 grayscale frame stacks;
- compact uint8 CPU replay;
- a convolutional dueling value/advantage head;
- Double-DQN targets;
- Huber loss and gradient clipping;
- a delayed target network;
- linearly annealed epsilon-greedy exploration;
- random and frozen snapshot opponents;
- periodic fixed-seed evaluation and best-checkpoint selection.

Generated checkpoints, CSV results, W&B runs, course notes, and reports are
excluded from version control. Training source code and tests are tracked.
