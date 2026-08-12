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
