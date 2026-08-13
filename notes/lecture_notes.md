# Integrated Course Notes / 课程综合笔记索引

This is the bilingual entry point for the course. The original English-only
integrated draft is preserved under `materials/legacy/lecture_notes_en.md` for
source traceability; it is no longer the recommended learning note.

这是课程笔记的双语总入口。旧版全英文综合草稿已原样保存在
`materials/legacy/lecture_notes_en.md`，用于追溯资料，但不再作为推荐阅读版本。

## Daily guides / 每日主讲义

| Day | Main note | Coverage / 内容 |
|---|---|---|
| Day 1 | `day-01/README.md` | ML, optimization, autodiff, neural networks, MDP, value/Q, DQN, actor-critic, PPO, evaluation / 机器学习、优化、自动微分、MDP、DQN、PPO 与评估 |
| Day 2 | `day-02/README.md` | stochastic games, cooperation/competition, Nash, non-stationarity, credit assignment, CTDE, Minecraft / 随机博弈、合作竞争、纳什均衡、MARL 难点、CTDE 与 Minecraft |
| Day 3 | `day-03/README.md` | imitation learning, MPE2, IQL/CQL, W&B, DQN/PyTorch / 模仿学习、MPE2 实践、实验追踪与 DQN 工程 |
| Day 4 | `day-04/README.md` | Boxing training, evaluation, and source comparison / Boxing 训练、评估与源码对照 |

## Supporting notes / 专题笔记

- `day-01/research-presentations.md`: course-team research talks / 助教与教师研究报告。
- `day-02/minecraft-agents.md`: Anssi's Minecraft lecture and reading map / Minecraft 演讲与论文路线。
- `day-03/imitation-learning.md`: complete IL lecture reconstruction / 模仿学习课堂还原。
- `day-03/imitation-learning-code-study.md`: supplied Alien code analysis / Alien 教学代码分析。
- `day-03/mpe2-practical.md`: MARL practical and W&B details / MARL 实践与实验记录。
- `day-03/dqn-breakout-pytorch.md`: PyTorch/CUDA and DQN implementation / PyTorch、CUDA 与 DQN。

## Learning sequence / 建议学习顺序

```text
loss and gradients
 -> neural function approximation
 -> MDP, return, value and Q
 -> DQN and PPO
 -> stochastic games and MARL
 -> demonstrations and imitation learning
 -> reproducible experiment tracking
 -> Boxing training and tournament evaluation
```

每一天都应按“直觉 → 正式定义 → 公式 → 数值例子 → 代码 → 失败模式 → 评估”阅读。只记住缩写不足以完成项目；能够解释 tensor shape、target 如何构造、数据由哪种 policy 产生，以及 evaluation 控制了哪些变量，才算真正理解。
