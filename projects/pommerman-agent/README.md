# Pommerman Agent

An original PyTorch training project for the four-player Pommerman Free-for-All
environment. The learning pipeline starts with behavioral cloning from the
built-in `SimpleAgent`, then fine-tunes one shared policy with PPO while rotating
through all four spawn positions.

这是一个独立实现的四人 Pommerman 智能体项目。训练分为两步：先模仿内置
`SimpleAgent`，让网络学会基本移动、放置炸弹与躲避；随后在四个出生位置轮换，
使用 PPO 对抗随机与规则智能体。

## Sources and Attribution

- [eugene/pommerman](https://github.com/eugene/pommerman) motivated the
  actor-critic direction and the BC-to-RL learning sequence.
- [MultiAgentLearning/playground](https://github.com/MultiAgentLearning/playground)
  provides the Apache-2.0 Pommerman environment used at runtime.

No source code from `eugene/pommerman` is copied here. That repository does not
publish an explicit software license. The model, encoder, BC trainer, PPO
trainer, evaluation code, tests, and scripts in this directory are the team's
independent implementation.

## Game Interface

FFA uses a fully observable `11 x 11` board and four agents. Each agent chooses
one of six actions per step:

| ID | Action | 中文 |
| --- | --- | --- |
| 0 | Stop | 原地等待 |
| 1 | Up | 向上移动 |
| 2 | Down | 向下移动 |
| 3 | Left | 向左移动 |
| 4 | Right | 向右移动 |
| 5 | Bomb | 放置炸弹 |

The encoder produces 16 spatial channels for walls, wood, flames, power-ups,
bomb timers and ranges, the learner, enemies, ammo, blast strength, kick
ability, and surviving-player count. An action mask prevents the policy from
selecting moves blocked by walls and prevents illegal bomb placement.

## Learning Pipeline

1. **Behavioral cloning / 行为克隆**
   Four `SimpleAgent` instances generate state-action demonstrations. The actor
   learns every expert action with cross-entropy. The safety action mask is
   applied during interaction, not while fitting expert labels.
2. **PPO fine-tuning / PPO 强化微调**
   The learner rotates through all four spawn slots and faces a 70/30 mixture
   of `SimpleAgent` and `RandomAgent` opponents. The official terminal reward
   remains the main objective. Small dense rewards encourage new cells,
   power-ups, valid bomb use, and enemy elimination.
3. **Evaluation / 评估**
   Deterministic evaluation reports return, episode length, and action usage
   separately for every spawn slot.

## Initial Baseline / 初始基线

The first checked run used 100 BC episodes followed by 100 PPO episodes. This
is a pipeline baseline rather than a finished competitive agent.

| Opponents | Games | Wins | Mean return |
| --- | ---: | ---: | ---: |
| Three `RandomAgent`s | 8 | 8 | +1.00 |
| Three `SimpleAgent`s | 4 | 0 | -1.00 |

初始模型已经稳定超过随机对手，但还不能击败规则智能体。样本量很小，因此这些
数字只证明训练、保存和评估链路有效，不能当作最终性能结论。下一阶段需要增加
PPO 对局数，并持续监控四个出生位、炸弹动作使用率和对 `SimpleAgent` 的胜率。

## WSL Setup

The official Playground still targets legacy Gym APIs, so this project uses a
separate Python 3.8 environment. It does not modify the Boxing or Combat Tank
environments.

```bash
cd /mnt/c/Users/gioia.zheng/Desktop/ai-and-computer-games/projects/pommerman-agent
bash scripts/setup_wsl.sh
```

The script pins the tested dependency set, installs CUDA-enabled PyTorch, clones
the official Playground at commit `5315f6d`, and installs it under the ignored
`.vendor/` directory.

## Train

```bash
conda activate pommerman
cd /mnt/c/Users/gioia.zheng/Desktop/ai-and-computer-games/projects/pommerman-agent

python -m src.train_bc \
  --episodes 100 \
  --device cuda \
  --checkpoint checkpoints/bc_policy.pt

python -m src.train_ppo \
  --episodes 1000 \
  --opponent mixed \
  --device cuda \
  --load checkpoints/bc_policy.pt \
  --checkpoint checkpoints/ppo_policy.pt
```

Or run both stages with `bash scripts/train_quickstart.sh`.

## Evaluate

```bash
python -m src.evaluate \
  --model checkpoints/ppo_policy.pt \
  --opponent simple \
  --games-per-role 5 \
  --device cuda
```

Rendering additionally requires a working OpenGL/GLU installation in WSL. It
is deliberately separate from headless training so missing GUI libraries do
not stop long runs.

## Tests

```bash
python -m pytest -q
```

Checkpoints, demonstration data, the cloned Playground source, and experiment
logs are local artifacts and are excluded from Git.
