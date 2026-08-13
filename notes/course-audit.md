# Course Coverage Audit / 课程笔记覆盖审计

## Audit rule / 审计标准

A topic is considered complete only when the notes contain more than a title:
English explanation, Chinese explanation, formal definition or equation where
appropriate, an intuitive or numerical example, implementation mapping, common
failure modes, and a connection to the Boxing project.

一个主题只有同时包含“中英文解释、正式定义/公式、例子、代码对应、失败模式和项目联系”时才标记为完整。老师尚未提供的文件会明确写成 unavailable，不使用推测内容冒充课堂资料。

## Day 1 / 第一天

| Topic / 主题 | Status / 状态 | Location / 位置 |
|---|---|---|
| Supervised, unsupervised, self-supervised, RL | Complete / 已补全 | `day-01/README.md` Sections 1, 7 |
| Linear regression, MSE, residual | Complete | Sections 2-3 + notebook |
| Logistic regression, sigmoid, BCE | Complete | Section 4 |
| Gradient descent and learning rate | Complete | Section 3 |
| Neural networks and function approximation | Complete | Section 5 |
| Automatic differentiation and backprop | Complete | Section 6 + survey source |
| Agent loop, trajectory, MDP, Markov property | Complete | Sections 7-8 |
| Reward, return, gamma, Goodhart | Complete | Section 9 |
| V, Q, Bellman equations | Complete | Section 10 |
| Q-learning and DQN | Complete | Section 11 |
| On/off-policy, offline RL | Complete | Section 12 |
| Model-free/model-based | Complete | Section 13 |
| Policy gradient, actor-critic, advantage | Complete | Section 14 |
| PPO clipping | Complete | Section 15 |
| Exploration and evaluation | Complete | Section 16 |
| Four research presentations | Complete with bilingual guide / 已有双语精读 | `day-01/research-presentations.md` |

## Day 2 / 第二天

| Topic / 主题 | Status / 状态 | Location / 位置 |
|---|---|---|
| Stochastic games and joint action | Complete | `day-02/README.md` Section 1 |
| Cooperative, zero-sum, general-sum | Complete | Section 2 |
| Partial observability and POSG | Complete | Section 3 |
| Joint policies and values | Complete | Section 4 |
| Best response, Nash, minimax | Complete | Section 5 |
| Non-stationarity and credit assignment | Complete | Section 6 |
| Coordination and scaling | Complete | Section 6 |
| Independent and centralized learning | Complete | Sections 7-8 |
| CTDE, parameter sharing, value decomposition | Complete | Sections 8-9 |
| Communication and opponent modelling | Complete | Section 10 |
| Minecraft environment evolution | Complete | `minecraft-agents.md` |
| VPT, IDM, pseudo-labels | Complete | Main Section 12 + specialist note |
| Hierarchy, memory, code-as-action | Complete | Main Section 13 + specialist note |
| Fuzzy tasks and proxy evaluation | Complete | Main Section 14 |
| Teacher's reading links | Indexed; named-only systems are not overclaimed | `minecraft-agents.md` Section 14 |

## Day 3 / 第三天

| Topic / 主题 | Status / 状态 | Location / 位置 |
|---|---|---|
| BC, distribution shift, cross-entropy | Complete | `day-03/README.md` Section 2 |
| IRL, reward family, feature expectation | Complete | Section 2 |
| GAIL, occupancy, discriminator reward | Complete | Section 2 |
| AIRL and shaping decomposition | Complete | Section 2 |
| Supplied Alien code and 512/530 shapes | Complete | Section 3 + code study |
| Missing expert demonstrations | Explicitly unavailable / 明确缺失 | material README |
| MPE2 Simple Spread and global reward | Complete | Sections 4-5 + practical |
| IQL, CQL, output counts, CTDE distinction | Complete | Section 5 |
| Actual IQL/CQL evaluation | Complete with limitations | Section 6 |
| W&B Models and Weave | Complete | Section 7 + practical |
| PyTorch/CUDA and Breakout DQN | Complete | Section 8 + DQN note |

## Day 4 / 第四天

| Topic / 主题 | Status / 状态 | Location / 位置 |
|---|---|---|
| Baseline source preservation | Complete | `materials/day-04/boxing-baseline/` |
| Double-Dueling DQN implementation | Complete | tracked `projects/boxing-dqn/src/` |
| Snapshot opponent curriculum | Complete | training source and Day 4 guide |
| Fixed-seed evaluation | Complete | `evaluate_boxing.py` |
| W&B offline/online logging | Complete | optional CLI mode |
| Unit and environment smoke tests | Complete | tracked tests + ignored results |
| Formal GPU training and report | Complete with honest non-convergence finding / 已完成并如实报告未稳定收敛 | Day 4 guide + ignored checkpoints/results/report |

## Known source limits / 已知资料限制

- The 2026 expert demonstration file for Alien was not supplied because it is too large. Full BC/IRL/GAIL reproduction cannot be claimed.
- The inspected instructor repository commit provides Boxing installation/environment instructions but no Boxing training source. The Day 3 baseline is our own preserved starting point.
- Several Minecraft systems were supplied only as reading links. The notes explain their place in the evolution but do not claim the complete architecture was presented in class.
- One training seed is a project result, not proof that an algorithm is universally superior. Multi-seed experiments remain recommended.

这些限制不是未完成的借口，而是结论边界。专业报告必须区分“已运行验证”“只做结构检查”“资料缺失无法复现”和“学习补充”。
