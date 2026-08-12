# Atari Boxing DQN

PettingZoo Atari Boxing training project with a compact DQN baseline, Breakout practice code, and CUDA-ready setup notes.

## Source Context

This repository is an independent project repository. The setup follows the instructor-provided course repository:

https://github.com/Hautamaki-lab/Summer-School-2026

It is not a fork of that repository.

Lecture notes, slides, notebooks, generated checkpoints, and local result files are intentionally excluded from version control.

## Files

```text
boxing_random.py       Random PettingZoo Boxing rollout
breakout_random.py     Random Gymnasium Breakout rollout
dqn_simple.py          Small Breakout DQN practice script
dqn_boxing.py          Boxing DQN training script
play_dqn_boxing.py     Render a saved Boxing DQN checkpoint
requirements.txt       Python package list
```

## Environment

The project was tested in WSL with a conda environment named `pettingzoo`.

```bash
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
python boxing_random.py
```

Random Breakout rollout:

```bash
python breakout_random.py
```

Small Boxing DQN smoke test:

```bash
python dqn_boxing.py --episodes 1 --max-steps 100 --learning-starts 20 --batch-size 8 --replay-size 500 --target-update 50 --save-every 1
```

Expected generated files:

```text
checkpoints/dqn_boxing.pt
results/dqn_boxing_training.csv
```

These files are ignored by git.

## Longer Run

```bash
python dqn_boxing.py --episodes 50 --max-steps 5000 --learning-starts 1000 --batch-size 32 --replay-size 5000 --target-update 1000 --save-every 10
```

After training:

```bash
python play_dqn_boxing.py --checkpoint checkpoints/dqn_boxing.pt --episodes 1
```
