# AI and Computer Games

Course projects and practical work for the University of Eastern Finland course
**Artificial Intelligence for Computer Games**.

## Course Source

The practical setup follows the instructor-provided repository:

- [Hautamaki-lab/Summer-School-2026](https://github.com/Hautamaki-lab/Summer-School-2026)
- [MARL IQL/CQL Tutorial](https://github.com/Hautamaki-lab/Summer-School-2026/blob/main/MARL-IQL-CQL-Tutorial.md)
- [Farama Foundation/MPE2](https://github.com/Farama-Foundation/MPE2)

This is an independent course-work repository, not a fork of the instructor repository.

## Repository Layout

```text
ai-and-computer-games/
|-- projects/
|   `-- boxing-dqn/               Atari Boxing DQN project
|-- practicals/
|   `-- mpe2-simple-spread/       Random, IQL, and centralized Q-learning
|-- final-work/                   Cooperative MARL final assignment
|-- notes/                        Local bilingual lecture notes
|-- notebooks/                    Local course notebooks
|-- materials/                    Local lecture PDFs and source material
`-- README.md
```

The repository tracks project code and documentation. Lecture notes, notebooks,
course materials, generated checkpoints, experiment results, and reports remain
available locally but are excluded from version control.

## Projects

### Atari Boxing DQN

```bash
cd projects/boxing-dqn
pip install -r requirements.txt
python examples/boxing_random.py
python src/dqn_boxing.py --episodes 1 --max-steps 100
```

See [projects/boxing-dqn/README.md](projects/boxing-dqn/README.md) for training,
CUDA, and playback commands.

### MPE2 Simple Spread

```bash
cd practicals/mpe2-simple-spread
python spread_random.py
python spread_iql.py
python spread_cql.py
python spread_evaluate.py
python wandb_ver.py --algorithm iql --wandb-mode offline
python weave_ver.py --algorithm iql --episodes 5
```

In this practical, `CQL` means **Centralized Q-Learning**, not Conservative
Q-Learning. The optional W&B entry point tracks training metrics, evaluation
tables, configuration, system utilization, and model artifacts. The Weave entry
point traces the call tree and trajectories of trained-policy evaluations.

### Final Work

```bash
cd final-work/code
pip install -r requirements.txt
python central_q_learning.py --episodes 8000 --seeds 5 --output-dir ../results
```

The final-work implementation trains a centralized tabular Q-learning agent in
a cooperative two-agent gridworld and generates evaluation tables and plots.

## Tested Environment

```text
Ubuntu via WSL
Python 3.12.13
PyTorch 2.13.0+cu126
CUDA 12.6
NVIDIA GeForce GTX 1650 Ti
MPE2 1.1.0
```
