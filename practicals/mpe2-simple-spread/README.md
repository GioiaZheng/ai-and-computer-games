# Simple Spread MARL Practice

This directory follows the University of Eastern Finland Summer School 2026 practical on multi-agent reinforcement learning.

## Sources and attribution

- Tutorial: [Hautamaki-lab/Summer-School-2026](https://github.com/Hautamaki-lab/Summer-School-2026), `MARL-IQL-CQL-Tutorial.md`
- Environment: [Farama Foundation/MPE2](https://github.com/Farama-Foundation/MPE2)
- Environment documentation: [MPE2 Documentation](https://mpe2.farama.org/)

MPE2 is maintained by the Farama Foundation and is derived from the original Multi-Agent Particle Environments. The exercise code in this directory was created from the course tutorial rather than presented as original environment or tutorial code.

## Files

```text
spread_random.py    Random-agent baseline
dqn.py              Shared DQN components
spread_iql.py       Independent Q-Learning
spread_cql.py       Centralized Q-Learning
spread_evaluate.py  Numerical and visual comparison
wandb_ver.py        IQL/CQL training with Weights & Biases tracking
checkpoints/        Generated model weights
```

In this tutorial, `CQL` means **Centralized Q-Learning**, not Conservative Q-Learning.

## Run in Ubuntu

```bash
conda activate pettingzoo
cd practicals/mpe2-simple-spread
python -m pip install -r requirements.txt

python spread_random.py
python spread_iql.py
python spread_cql.py
python spread_evaluate.py
```

Simple Spread returns negative distance-based team rewards. A value closer to zero is better.

## Weights & Biases version

`wandb_ver.py` keeps the original tutorial scripts unchanged and adds experiment
configuration, live metrics, final evaluation, and model artifacts.

- Platform: [wandb.ai](https://wandb.ai/)
- Experiment logging documentation: [Log objects and media](https://docs.wandb.ai/models/track/log)
- Model versioning documentation: [W&B Artifacts](https://docs.wandb.ai/models/artifacts)

Log in once before using online mode:

```bash
wandb login
```

Train IQL or centralized Q-learning:

```bash
python wandb_ver.py --algorithm iql
python wandb_ver.py --algorithm cql
```

Use offline mode when the classroom network is unavailable:

```bash
python wandb_ver.py --algorithm iql --wandb-mode offline
wandb sync wandb/offline-run-*
```

Short smoke test without creating a W&B run:

```bash
python wandb_ver.py \
  --algorithm cql \
  --total-timesteps 200 \
  --learning-starts 20 \
  --batch-size 8 \
  --buffer-size 500 \
  --evaluation-games 5 \
  --wandb-mode disabled
```

Tracked values include episode return, rolling 100-episode mean return,
epsilon, DQN loss, replay-buffer size, evaluation mean and standard deviation,
run configuration, and model checkpoints as W&B artifacts.
