# Day 3 Complete Guide / 第三天完整讲解

## 0. What belongs to Day 3? / Day 3 包含什么？

Day 3 continues **Multi-Agent Reinforcement Learning (MARL，多智能体强化学习)**,
introduces **Imitation Learning (IL，模仿学习)**, and then turns the theory into
an MPE2 practical. The Atari teaching code, W&B scripts, and compact Breakout
DQN received alongside the class are included here as implementation studies.

第三天并不是几个互不相关的主题拼在一起。上午继续讨论 **MARL（多智能体强化学习）**，重点是多个智能体如何在共享环境中学习；随后进入 **IL（模仿学习）**，研究没有明确奖励或奖励很难设计时，如何从专家示范中学习；下午再用 MPE2 的 Simple Spread 把“共享奖励、局部观察、独立决策、联合决策”真正落实到代码。老师提供的 Atari Alien 模仿学习代码、W&B 版本，以及后来收到的 Breakout DQN，都属于这条实践链上的实现材料。

```text
Morning theory / 上午理论
  MARL continuation
  Imitation Learning: BC -> IRL -> GAIL -> AIRL

Code study / 代码研究
  Atari Alien demonstrations
  BC encoder and pretrained weights
  learned rewards with IRL and GAIL

Afternoon practical / 下午实践
  MPE2 Simple Spread
  random baseline -> IQL -> centralized Q-learning
  common-seed evaluation

Experiment engineering / 实验工程
  W&B metrics, artifacts, and Weave traces
  PyTorch + CUDA verification
  compact Breakout DQN -> maintained Boxing DQN
```

This page is the reading route and conceptual bridge. The detailed derivations,
code walkthroughs, photographs, results, and caveats remain in four companion
notes:

这份文件是 **可以从头独立阅读的 Day 3 主讲义**。它负责解释概念之间的因果关系，并给出足够的公式、例子和代码映射；四份专题笔记则保留更完整的课堂照片还原、逐行代码分析、运行结果和工程注意事项。第一次学习建议先读本页，真正动手时再进入专题页。

| Read in this order | Detailed note / 专题笔记 | Main purpose / 作用 |
|---:|---|---|
| 1 | `imitation-learning.md` | Complete BC, IRL, GAIL, and AIRL lecture from live photos plus the 2024/2026 slides |
| 2 | `imitation-learning-code-study.md` | Supplied Alien code and pretrained BC weights, tensor shapes, and implementation caveats |
| 3 | `mpe2-practical.md` | MPE2, global reward, IQL, centralized Q-learning, outputs, results, W&B, and Weave |
| 4 | `dqn-breakout-pytorch.md` | PyTorch/CUDA setup and formula-to-code study of the supplied Breakout DQN |

## 1. One unifying question / 一条贯穿主线

All Day 3 topics ask how several sources of information become a useful policy:

- **Environment reward / 环境奖励:** DQN, IQL, and centralized Q-learning learn
  from temporal-difference targets.
- **Expert action / 专家动作:** BC directly predicts what the expert would do.
- **Expert behavior distribution / 专家行为分布:** GAIL rewards behavior that a
  discriminator considers expert-like.
- **Inferred objective / 推断出的目标:** IRL and AIRL try to explain why the
  expert behaves that way.
- **Team outcome / 团队结果:** cooperative MARL gives multiple agents a shared
  global reward.

换成更直白的中文，Day 3 一直在回答同一个问题：**智能体到底根据什么信号，判断刚才的行为值得保留？**

- 环境直接给分时，DQN、IQL、集中式 Q-learning 用 TD target 学习。
- 专家直接告诉“这个状态该做什么”时，BC 把问题变成监督分类。
- 只有专家轨迹、没有可靠奖励时，IRL 尝试反推出专家在乎什么。
- 不要求解释专家动机，只要求整体行为像专家时，GAIL 用判别器产生代理奖励。
- 多个智能体合作时，同一个全局奖励必须被分配到不同智能体、不同时间步的决策上。

The common skeleton is:

\[
\text{data or interaction}
\longrightarrow \text{learning signal}
\longrightarrow \text{model update}
\longrightarrow \text{policy}
\longrightarrow \text{evaluation}.
\]

### 1.1 Prerequisite concepts / 必备前置概念

这些词在后文反复出现。只记中文翻译不够，还要知道每个对象在程序里是什么。

| Concept | Formal meaning | 中文理解 | Code-level example |
|---|---|---|---|
| **Agent** | decision maker | 智能体，即负责选择动作的学习者 | `agent_0`, `agent_1` |
| **Environment** | transition and reward generator | 环境，接收动作后产生下一状态与奖励 | MPE2 `env.step(actions)` |
| **State** \(s_t\) | complete task information | 完整状态，理论上足以预测下一步 | 两个智能体和两个地标的全部位置 |
| **Observation** \(o_{i,t}\) | information visible to agent \(i\) | 局部观察，可能只是状态的一部分 | `observations["agent_0"]` |
| **Action** \(a_t\) | a decision applied to the environment | 动作 | 不动、上下左右之一 |
| **Joint action** \(\mathbf a_t\) | actions of all agents together | 联合动作，所有智能体动作的组合 | `(a_0, a_1)` |
| **Reward** \(r_t\) | immediate scalar feedback | 当前一步得到的即时分数 | Simple Spread 的负距离 |
| **Return** \(G_t\) | discounted future reward sum | 从当前开始的累计长期回报 | \(G_t=\sum_{k=0}^{T-t}\gamma^k r_{t+k}\) |
| **Policy** \(\pi\) | mapping from information to action probabilities | 策略，决定看到什么时做什么 | epsilon-greedy 或神经网络 softmax |
| **Q-value** \(Q(s,a)\) | expected return after action \(a\) | 在状态中做某动作后，预期能获得多少长期回报 | Q-network 的一个输出 |
| **Trajectory** \(\tau\) | an ordered interaction sequence | 轨迹，一整个 episode 的时序记录 | \((s_0,a_0,r_0,s_1,\ldots)\) |

Reward 和 return 很容易混淆。`reward` 是“一步得分”，而 `return` 是“从现在往后的总成绩”。一个动作可能立即得到小的负奖励，却让智能体以后避开碰撞、覆盖地标，因此长期 Q-value 仍然很高。

The distinction between **state** and **observation** becomes crucial in MARL. The environment may have a complete state \(s\), while agent \(i\) acts only from \(o_i\). Two agents can therefore be in the same world but receive different network inputs. / 在多智能体任务里，环境拥有完整状态，但每个智能体可能只看到自己的局部观察。所以“两个智能体处于同一个世界”不意味着“两个 Q-network 得到相同输入”。

### 1.2 From immediate reward to a Q-value / 从即时奖励到 Q 值

Q-value is not a confidence score and not the probability of choosing an action. It estimates long-term discounted return:

\[
Q^\pi(s,a)
=\mathbb E_\pi\left[
\sum_{k=0}^{\infty}\gamma^k r_{t+k}
\mid s_t=s,a_t=a
\right].
\]

中文拆解：先固定“当前状态是 \(s\)、当前动作是 \(a\)”，之后继续按照策略 \(\pi\) 行动；把未来每一步奖励按 \(1,\gamma,\gamma^2,\ldots\) 衰减后求和，再对环境随机性取平均。Q-network 输出的每个数字都在估计这个量。

例如 \(\gamma=0.9\)，某动作立即奖励为 \(-1\)，但下一步开始预计获得 \(5\) 的回报，那么粗略 target 是

\[
y=-1+0.9\times5=3.5.
\]

所以“当前奖励是负数”不代表动作一定不好。RL 之所以需要 Bellman target，就是为了把未来影响传播回当前决策。

## 2. Imitation Learning / 模仿学习

### 2.1 RL versus IL / 强化学习与模仿学习

**Reinforcement Learning (RL，强化学习)** learns from reward supplied by an
environment:

\[
(s_t,a_t,r_t,s_{t+1}) \longrightarrow \text{policy improvement}.
\]

**Imitation Learning (IL，模仿学习)**, also called **Learning from
Demonstration (LfD，从示范中学习)**, learns from trajectories produced by an
expert:

\[
\tau_E=(s_0,a_0,s_1,a_1,\ldots,s_T).
\]

The expert is the demonstrator; the learner is the policy being trained. The
important distinction is not that IL has no objective, but that the objective
is obtained from demonstrations rather than only from a hand-written
environment reward.

模仿学习的“专家”不一定是人类顶尖玩家，也可以是规则程序、旧版本模型、搜索算法，甚至多种策略组成的数据集。专家质量决定了性能上限：如果示范里持续包含错误，BC 会认真复制这些错误，GAIL 也可能把错误当成“专家风格”。因此 IL 的第一步往往不是选模型，而是检查示范覆盖范围、动作含义、episode 边界和数据质量。

**A concrete Boxing example / Boxing 直觉例子：** 环境只在击中对手时给稀疏奖励，随机探索很难学会“先靠近、保持合适距离、再出拳”的动作序列。人类示范直接提供这一序列。BC 学“画面到按键”的映射；IRL 猜测专家可能在奖励“接近有效攻击距离、避免被击中”；GAIL 则不解释原因，只要求学习者产生与专家相似的状态动作分布。

### 2.2 Behavioral Cloning (BC，行为克隆)

BC asks: **What action should I take in this state?** It treats expert
state-action pairs as supervised classification data:

\[
\mathcal D_E=\{(s_i,a_i)\}_{i=1}^{N},
\qquad
\pi_\theta(a\mid s)=\operatorname{softmax}(f_\theta(s)).
\]

For discrete actions, the cross-entropy objective is

\[
\mathcal L_{BC}(\theta)
=-\frac{1}{N}\sum_{i=1}^{N}
\log \pi_\theta(a_i\mid s_i).
\]

Training is straightforward: load a batch, encode each observation, produce
action logits, compare them with expert labels, backpropagate, and update the
network. Its central failure mode is **distribution shift（分布偏移）**. A small
mistake can move the learner into a state absent from the expert dataset, after
which further mistakes compound.

中文逐步理解：一张游戏画面是输入，专家按下的按钮是分类标签。假设动作空间有 18 个动作，网络输出 18 个 logits；softmax 把它们变成概率，cross-entropy 提高专家动作的概率。它和普通图片分类的数学形式相同，区别在于样本来自连续轨迹，相邻样本高度相关，而且一个分类错误会改变下一张画面。

**Worked example / 数值例子：** 对某个状态，网络给三个动作概率

\[
\pi_\theta(\cdot\mid s)=[0.70,0.20,0.10].
\]

若专家动作是第一个，单样本损失为 \(-\log0.70\approx0.357\)；若专家动作是第三个，损失为 \(-\log0.10\approx2.303\)。后者惩罚更大，梯度会更强地推动第三个动作概率上升。

**Why accuracy can lie / 为什么分类准确率会骗人：** 假设验证集动作准确率为 95%，并不意味着玩 200 步时有 95% 概率成功。若粗略假设每一步独立且都必须正确，连续 200 步无错误的概率只有

\[
0.95^{200}\approx3.5\times10^{-5}.
\]

真实误差并不独立，但这个计算说明了长时序任务为何会放大微小错误。部署时更应报告 episode return、胜率、恢复能力，而不只报告离线 action accuracy。

**Common misconception / 常见误区：** BC 是 offline learning，并不表示训练出的 policy 永远不与环境交互；它表示参数更新只需要固定示范数据。最终评估仍然应把 policy 放回环境运行。

### 2.3 Inverse Reinforcement Learning (IRL，逆强化学习)

Standard RL starts with a reward and searches for a policy:

\[
R \longrightarrow \pi^*.
\]

IRL starts with expert behavior and searches for a reward that explains it:

\[
\pi_E \longrightarrow \hat R_E
\longrightarrow \text{RL} \longrightarrow \pi.
\]

The researcher must still choose a reward representation, called the
"skeleton" in the lecture. A common linear family is

\[
R_w(s,a)=w^\top\phi(s,a)
=\sum_{k=1}^{K}w_k\phi_k(s,a),
\]

where \(\phi_k\) is a meaningful feature and \(w_k\) is its learned importance.
For a policy \(\pi\), its discounted feature expectation is

\[
\mu(\pi)
=\mathbb E_{\tau\sim\pi}
\left[\sum_{t=0}^{T}\gamma^t\phi(s_t,a_t)\right].
\]

Feature-matching IRL tries to make

\[
\mu(\pi)\approx\mu(\pi_E).
\]

The outer loop updates reward weights; the inner loop trains a new RL policy
under the current learned reward. This nested optimization is why IRL can be
slow. Reward recovery is also non-identifiable: several rewards may explain
the same expert policy.

中文直觉：BC 看见专家往右走，就学“这里按右”；IRL 会进一步问“为什么往右？”可能是为了靠近对手，也可能是为了远离边界。我们先规定奖励由哪些特征组成，再学习每个特征的权重。这里的 `skeleton` 不是答案，而是允许模型搜索答案的语言。如果没有“与对手距离”这个特征，线性 IRL 就不可能凭空恢复出“保持攻击距离”这一目标。

**Worked feature example / 特征例子：** 定义 Boxing 特征

\[
\phi(s,a)=
\begin{bmatrix}
\text{hit opponent}\\
\text{get hit}\\
\text{distance in attack range}
\end{bmatrix},
\qquad
w=
\begin{bmatrix}
2.0\\-3.0\\0.5
\end{bmatrix}.
\]

若某一步特征为 \([1,0,1]^\top\)，奖励是 \(2.0+0+0.5=2.5\)；若为 \([0,1,0]^\top\)，奖励是 \(-3.0\)。IRL 学的是 \(w\)，但特征 \(\phi\) 的设计已经表达了研究者认为哪些事件可能重要。

**Why reward is ambiguous / 为什么奖励不可唯一确定：** 若把所有奖励乘以正数，最优策略往往不变；加入某些 potential-based shaping 项也可能不改变最优行为。因此 IRL 恢复的通常是“能解释行为的一类奖励”，不是专家大脑中唯一真实的心理目标。

### 2.4 GAIL / 生成对抗模仿学习

**Generative Adversarial Imitation Learning (GAIL，生成对抗模仿学习)** maps the
GAN idea onto trajectories:

- expert transitions are the real samples;
- the policy is the generator;
- the discriminator distinguishes expert from learner state-action pairs.

A common discriminator objective is

\[
\max_D\;
\mathbb E_{(s,a)\sim\rho_E}[\log D(s,a)]
+\mathbb E_{(s,a)\sim\rho_\pi}[\log(1-D(s,a))],
\]

where \(\rho_E\) and \(\rho_\pi\) are expert and learner occupancy measures.
Depending on the label convention, policy reward is commonly written as

\[
r_D(s,a)=\log D(s,a)
\quad\text{or}\quad
r_D(s,a)=-\log(1-D(s,a)).
\]

The formula must match what \(D=1\) means in the implementation. GAIL avoids a
separate explicit reward-recovery stage, but adversarial training can be
unstable and repeatedly requires fresh environment interaction.

中文训练循环可以分为两个交替的学习问题：

1. 固定 policy，用专家样本标记为 1、学习者样本标记为 0，训练 discriminator。
2. 固定 discriminator，把“看起来像专家”的程度变成 reward，再用 PPO 等 RL 算法更新 policy。
3. 新 policy 会访问新状态，因此必须重新采样轨迹并继续训练 discriminator。

**Occupancy measure / 占用分布：** GAIL 不只比较某一时刻动作是否相同，而是比较长期来看哪些 \((s,a)\) 经常出现。两个 policy 可以在同一状态选择不同的等价动作，也可以用不同路径到达相似结果；occupancy matching 比逐帧复制更关注整体行为分布。

**Reward-label example / 标签方向例子：** 如果 \(D(s,a)=0.9\) 表示“90% 像专家”，那么 \(\log D\approx-0.105\)，而 \(-\log(1-D)\approx2.303\)，两者都会相对鼓励更大的 \(D\)，但尺度和梯度不同。如果代码把 1 定义为 learner，却仍照抄公式，reward 方向就会完全反转。

**Why unstable / 为什么不稳定：** policy 和 discriminator 同时变化。判别器过强时，所有 learner 样本都被轻易判为假，policy 获得的有效梯度可能很弱；判别器过弱时，reward 又没有区分度。这和普通监督学习面对固定标签不同。

### 2.5 AIRL / 对抗式逆强化学习

**Adversarial Inverse Reinforcement Learning (AIRL，对抗式逆强化学习)** gives
the discriminator a structure intended to separate reward from reward shaping:

\[
D_{\theta,\varphi}(s,a,s')
=\frac{\exp(f_{\theta,\varphi}(s,a,s'))}
{\exp(f_{\theta,\varphi}(s,a,s'))+\pi(a\mid s)},
\]

\[
f_{\theta,\varphi}(s,a,s')
=g_\theta(s,a)+\gamma h_\varphi(s')-h_\varphi(s).
\]

Here \(g_\theta\) approximates the underlying reward, while \(h_\varphi\) is a
potential-based shaping or value-like term. GAIL mainly seeks expert-like
behavior; AIRL additionally aims for a reward structure that can be recovered
and transferred.

中文理解：GAIL 的 discriminator 分数更像“模仿得像不像”的考试成绩，这个成绩不一定能直接告诉我们任务真正奖励什么。AIRL 特意把判别器拆成两个角色：\(g_\theta\) 尝试描述任务本身，\(h_\varphi\) 吸收为了加速学习而存在的 shaping 信息。理想情况下，换一个环境动力学后，任务奖励 \(g_\theta\) 比单纯的判别器分数更有迁移价值。

**GAIL versus AIRL example / 对比例子：** 专家总是沿墙走，可能只是训练环境中沿墙最容易导航。GAIL 可能把“靠墙”本身当成专家特征；AIRL 希望把“到达目标”放进可恢复 reward，把“沿墙”视为特定环境下的 value/shaping 结果。但这种分离依赖模型结构和理论假设，并非自动保证成功。

### 2.6 How to choose among BC, IRL, GAIL, and AIRL? / 如何选择？

| Situation / 情况 | First method to try / 首选 | Why / 原因 |
|---|---|---|
| Large fixed demonstration dataset, no simulator access / 大量固定示范、不能访问环境 | BC | Only offline supervised updates are required / 只需离线监督学习 |
| Reward itself must be interpreted or transferred / 需要解释或迁移奖励 | IRL or AIRL | They explicitly model a reward / 显式学习奖励结构 |
| Simulator is available and behavior matching matters most / 可反复交互、主要追求像专家 | GAIL | Matches occupancy without manually specifying reward features / 无需手写完整奖励特征 |
| Need a fast sanity-check baseline / 先验证数据是否正确 | BC | Simplest data and optimization contract / 数据与训练流程最简单 |

Professional practice usually starts with BC even when a more advanced method is planned. If BC cannot learn from the demonstrations, GAIL or IRL will not magically repair misaligned actions, broken episode boundaries, or incorrect preprocessing. / 专业实践中，即使最终目标是 GAIL 或 IRL，也通常先做 BC。因为 BC 是检查数据管线最便宜的工具；如果状态和动作根本没对齐，更复杂的方法只会让错误更隐蔽。

## 3. Supplied Alien implementation / 收到的 Alien 教学代码

The Day 3 material includes `UEF Imitation Learning.pdf`, `bc_agent.py`,
`irl_agent.py`, `gail_agent.py`, `utils.py`, and `bc_model_example.pt`. The
large expert-demonstration file was not supplied. Originals are preserved under
`materials/day-03/imitation-learning-2026/`.

这些文件不能只看“网络有几层”，更重要的是检查 **data contract（数据契约）**：每个函数期望什么形状、dtype 和语义。模仿学习中最危险的错误往往不会报异常，例如把 \(s_{t+1}\) 与 \(a_t\) 配对，张量形状仍然合法，但模型学到的是错位关系。

The shared implementation flow is:

```text
210 x 160 RGB Atari observation
  -> resize to 64 x 64
  -> grayscale
  -> CNN encoder
  -> 512-dimensional feature vector
```

- **BC** adds an 18-action classification head and minimizes cross-entropy.
- **IRL** removes the BC action head, uses the 512 encoder features, learns
  feature weights, wraps the environment with the learned reward, and trains PPO.
- **GAIL** concatenates 512 state features with an 18-dimensional one-hot
  action, so the discriminator input has \(512+18=530\) values.

形状可以沿着网络逐步追踪：

```text
one frame                  (64, 64)
add channel dimension      (1, 64, 64)
add batch dimension        (B, 1, 64, 64)
CNN encoder output         (B, 512)
BC action logits           (B, 18)
one-hot action for GAIL    (B, 18)
GAIL concatenated input    (B, 530)
discriminator output       (B, 1)
```

`B` 是 batch size。Batch 维度必须一直保留，因为 loss 通常对一个 mini-batch 求平均。`logits` 是 softmax 前的原始分数，不是概率；`one-hot action` 则是长度 18、只有专家动作位置为 1 的向量。

The pretrained `.pt` file is the complete BC state dictionary, not only an
encoder. It was inspected successfully. The Python files pass syntax checks,
but full training was not run because demonstrations are absent and first-time
Atari/ROM setup was explicitly not recommended during class.

The code is educational rather than production-ready. Important checks before
reuse include trajectory boundaries, \(\gamma^t\) discounting, state-action
alignment, tensor device consistency, scalar reward types, paired GAIL
sampling, held-out evaluation, and reproducible seeds.

这些检查为什么重要：

- **Trajectory boundary / 轨迹边界：** 不能把一个 episode 末尾和下一个 episode 开头当成连续状态，否则 return 和 feature expectation 都会跨局累计。
- **Discounting / 折扣：** feature expectation 应让第 \(t\) 步乘 \(\gamma^t\)，而不是把所有帧当成同等重要的无序图片。
- **State-action alignment / 状态动作对齐：** discriminator 训练与产生 reward 时必须对同一种 \((s_t,a_t)\) 语义评分。
- **Device / 设备：** CPU tensor 与 CUDA model 混用会直接报错；更隐蔽的是频繁 CPU/GPU 拷贝让训练极慢。
- **Termination / 终止：** `terminated` 表示任务自然结束，`truncated` 表示时间上限等外部截断；两者都通常结束 episode，但在 bootstrap 语义上可能需要区分。
- **Held-out evaluation / 留出评估：** 训练集 action accuracy 只能说明记住示范，不能说明能在环境中稳定行动。

**What can be verified without the large demos? / 没有专家数据还能验证什么？** 可以加载 `bc_model_example.pt`、检查参数名与形状、输入一张符合预处理契约的假图像并验证输出 18 个 logits。这证明模型文件和网络结构兼容，但不能证明策略表现优秀。模型“能 forward”与“会玩游戏”是两个不同结论。

## 4. MPE2 Simple Spread practical / MPE2 Simple Spread 实践

The practical follows the course repository's `MARL-IQL-CQL-Tutorial.md` and
uses [MPE2](https://mpe2.farama.org/) from the
[Farama Foundation](https://github.com/Farama-Foundation/MPE2).

The classroom configuration is:

```python
env = simple_spread_v3.parallel_env(
    N=2,
    local_ratio=0.0,
    max_cycles=25,
    continuous_actions=False,
)
```

There are two agents, two landmarks, and five discrete actions per agent.
Observations are numeric vectors, not images. With `local_ratio=0.0`, both
agents receive the same **global reward（全局奖励）**.

这里的“粒子”可以理解成二维平面上的小圆点。两个 agent 要分别靠近两个 landmark，同时避免碰撞。每个 agent 的 observation 通常包含自己的速度与位置、各 landmark 相对自己的位置、其他 agent 相对自己的位置。因为相对坐标的参考点不同，`agent_0` 与 `agent_1` 的 observation 数值不同，即使它们处在同一个全局场景中。

One conceptual form of the Simple Spread team reward is

\[
r_{global}(s)
=-\sum_{\ell\in\mathcal L}
\min_{i\in\{1,\ldots,N\}}
\lVert p_i-p_\ell\rVert_2,
\]

possibly with collision penalties according to the environment version. Each
landmark contributes the distance to its nearest agent. The team improves when
landmarks are collectively covered, rather than when every agent independently
chases the same target.

**Worked reward example / 奖励数值例子：** 假设两个 landmark 到最近 agent 的距离分别是 \(0.2\) 和 \(0.7\)，先忽略碰撞惩罚：

\[
r_{global}=-(0.2+0.7)=-0.9.
\]

如果下一步两个距离变成 \(0.1\) 和 \(0.3\)，奖励变成 \(-0.4\)。虽然仍是负数，但 \(-0.4>-0.9\)，表示团队状态变好了。这个环境的目标不是追求某个神秘的正分，而是让负距离尽量接近 0。

为什么对每个 landmark 取 `min`？因为一个 landmark 只需要最近的 agent 覆盖。若两个 agent 都挤向同一个 landmark，另一个 landmark 的最近距离仍然很大，团队奖励就不会好。这个公式因此间接鼓励分工。

The reward mixture can be understood as

\[
r_i=\lambda r_{local,i}+(1-\lambda)r_{global},
\]

where `local_ratio` is \(\lambda\). At \(\lambda=0\), all learning is driven by
the shared team signal. This encourages cooperation but creates the
**credit-assignment problem（信用分配问题）**: an agent cannot directly tell how
much of a reward change came from its own action.

**Credit assignment / 信用分配再解释：** 假设 `agent_0` 正确走向左侧 landmark，而 `agent_1` 撞到了队友，全局奖励变差。两个 replay buffer 都会保存同一个较差 reward。`agent_0` 无法从这一条样本直接知道“我的动作其实有帮助，问题出在队友”。经过大量不同组合的样本后，Q-network 才可能统计性地分离这些影响。

全局奖励并不是总比局部奖励好。纯局部奖励容易让每个 agent 只顾自己；纯全局奖励又让责任归因困难。`local_ratio` 提供的是目标设计选择，不是普通的训练超参数。改变它等于改变“什么叫成功”。

## 5. IQL and centralized Q-learning / IQL 与集中式 Q 学习

### 5.1 Independent Q-Learning (IQL，独立 Q 学习)

Each agent treats the other learner as part of its environment and owns a
separate network:

\[
Q_0(o_0,a_0;\theta_0),
\qquad
Q_1(o_1,a_1;\theta_1).
\]

For agent \(i\), the DQN target is

\[
y_i=r_i+\gamma(1-d_i)
\max_{a_i'}Q_i^-(o_i',a_i';\theta_i^-),
\]

and the loss is

\[
\mathcal L_i
=\frac1B\sum_{b=1}^{B}
\operatorname{Huber}
\left(y_{i,b}-Q_i(o_{i,b},a_{i,b};\theta_i)\right).
\]

Although both agents receive the same global reward, they see different
observations and update different Q-networks. The environment is non-stationary
from each learner's perspective because its teammate's policy also changes.

中文逐步流程：

1. `agent_0` 把 \(o_0\) 输入自己的 Q-network，得到 5 个 Q 值并选出 \(a_0\)。
2. `agent_1` 把不同的 \(o_1\) 输入另一个 Q-network，得到另一组 5 个 Q 值并选出 \(a_1\)。
3. 环境同时执行 `{agent_0: a_0, agent_1: a_1}`。
4. 两个 learner 各自保存 \((o_i,a_i,r_{global},o_i',d_i)\)。
5. 两个 optimizer 分别更新，不共享参数；相同 reward 不会让两个网络变成同一个网络。

**Worked TD example / TD target 数值例子：** 假设 \(\gamma=0.95\)，当前共享奖励 \(r=-0.4\)，target network 对下一观察的最大 Q 值为 \(2.0\)，且 episode 未结束：

\[
y=-0.4+0.95\times2.0=1.5.
\]

如果在线网络对已执行动作预测 \(Q(o,a)=1.1\)，TD error 是

\[
\delta=y-Q(o,a)=1.5-1.1=0.4.
\]

优化会推动该动作 Q 值向 1.5 靠近。若 episode 已结束，\((1-d)=0\)，target 只剩当前 reward，不能从不存在的下一状态 bootstrap。

**Non-stationarity / 非平稳性：** 在单智能体 MDP 中，固定动作后环境转移规律通常不随训练改变；在 IQL 中，队友策略不断更新，因此“我做右移后通常会发生什么”也在变化。Replay buffer 里旧样本来自旧队友，当前样本来自新队友，这会削弱普通 DQN 的平稳性假设。

### 5.2 Centralized Q-Learning (CQL，集中式 Q 学习)

In this tutorial, **CQL means Centralized Q-Learning**, not Conservative
Q-Learning from offline RL. It concatenates observations and represents an
action pair as one joint action:

\[
o_{joint}=[o_0;o_1],
\qquad
a_{joint}=(a_0,a_1).
\]

The centralized target is

\[
y=r_{global}+\gamma(1-d)
\max_{a'_{joint}}
Q_{central}^-(o'_{joint},a'_{joint}).
\]

If each agent has \(m=5\) actions and there are \(N=2\) agents, then

\[
|\mathcal A_{joint}|=m^N=5^2=25.
\]

One encoding is

\[
a_{joint}=a_0\cdot5+a_1,
\qquad
a_0=\left\lfloor\frac{a_{joint}}5\right\rfloor,
\qquad
a_1=a_{joint}\bmod5.
\]

For example, joint index 17 decodes to `(3, 2)`.

集中式网络直接看见 \([o_0;o_1]\)，因此可以学习“当 agent 0 在左边、agent 1 在右边时，动作组合 `(left, right)` 很好”。IQL 的两个网络只能各自评估单个动作；协调信息只能通过共享 reward 间接传回来。

**Joint-value example / 联合动作值例子：** 假设对某个 joint observation，网络部分输出为：

```text
Q(joint_obs, (0, 0)) = -3.0   both stay still
Q(joint_obs, (1, 1)) = -2.4   both move to the same side
Q(joint_obs, (1, 3)) = -0.6   agents separate toward two landmarks
```

即使 `action 1` 对两个 agent 单独看来都像合理动作，联合组合 `(1,1)` 可能导致拥挤。集中式 Q 值可以直接区分 `(1,1)` 与 `(1,3)` 的团队效果。

### 5.3 Exactly how many results? / 最后到底有多少个结果？

| Method | Networks | Q-values produced | Argmax results | Environment actions |
|---|---:|---:|---:|---:|
| IQL | 2 | \(2\times5=10\) total | 2, one per network | 2 |
| Centralized Q-learning | 1 | \(5^2=25\) joint values | 1 joint index | 2 after decoding |

The 10 IQL values are two unrelated sets of five individual-action values. The
25 centralized values are one set containing every action pair. For \(N\)
agents with \(m\) actions each, centralized output width grows as \(m^N\), the
**combinatorial explosion（组合爆炸）**. This tutorial centralizes both training
and execution; it is not yet CTDE because execution still needs all agents'
observations.

“最后有多少个结果”要分三层回答：

- **网络原始输出：** IQL 总共算出 10 个数；集中式网络算出 25 个数。
- **argmax 决策结果：** IQL 得到两个 index；集中式网络只得到一个 joint index。
- **送入环境的动作：** 两者最终都必须提供两个动作，因为环境里有两个 agent。

例如 CQL 输出 25 个 Q 值后，`argmax=17` 只是一个编码结果；解码后才得到两个真实动作 `(3,2)`。所以“CQL 只输出一个动作”是不准确的，它输出的是一个代表动作组合的 index。

**Scaling example / 扩展性例子：** 每个 agent 仍有 5 个动作时，2 个 agent 需要 25 个 joint Q 值，3 个需要 125 个，6 个需要 \(5^6=15,625\) 个。输入 observation 只是线性拼接，输出动作组合却指数增长。这就是简单集中式枚举无法扩展到大型 MARL 的根本原因之一。

### 5.4 IQL, centralized control, and CTDE / IQL、集中控制与 CTDE

**CTDE (Centralized Training with Decentralized Execution，集中训练、分散执行)** means extra global information may be used during training, but each deployed agent can act from its own local observation. / CTDE 的关键不是“训练时有一个大网络”，而是执行时每个 agent 不需要把全部 observation 发给中央控制器。

- IQL naturally executes independently, but it does not exploit centralized training information.
- This tutorial's centralized Q-network uses joint observation both in training and execution, so it is centralized control, not CTDE.
- Methods such as value-decomposition or centralized-critic approaches seek better coordination during training while retaining decentralized policies for execution.

为什么需要 CTDE？真实机器人可能通信受限，游戏角色也可能只能看到局部视野。训练服务器可以暂时访问完整 state 来改善 credit assignment，但比赛运行时必须遵守每个 agent 的信息限制。

## 6. Practical results / 实践结果

Random, IQL, and centralized Q-learning were evaluated on the same 100 unseen
seeds after 50,000 environment steps of training for each learned method:

| Method | Mean team return | Standard deviation | Improvement over random |
|---|---:|---:|---:|
| Random | -40.361 | 12.750 | 0.000 |
| IQL | -17.236 | 6.291 | 23.125 |
| Centralized Q-learning | -16.745 | 8.674 | 23.615 |

The comparison is

\[
\Delta_{CQL-IQL}
=-16.745-(-17.236)
\approx0.491.
\]

Both learned policies clearly beat random in this run. Centralized Q-learning
has the highest mean but also higher variance than IQL, and a \(0.491\) mean
advantage from one training seed is not evidence of universal superiority.
Multiple training seeds and confidence intervals are needed for a stronger
claim.

中文解读不能只看“谁的数字最大”：

- Random 平均为 \(-40.361\)，IQL 与 CQL 都提高了约 23 分，说明两种学习方法确实学到了覆盖 landmark 的行为。
- CQL 均值比 IQL 高 \(0.491\)，但 CQL 标准差也更大，表示不同初始布局下表现波动更明显。
- 这 100 个 evaluation seed 衡量的是 **同一个已训练模型面对不同初始局面** 的差异，并不包含“重新训练模型时随机初始化带来的差异”。
- 要比较算法稳定性，应使用多个 training seed：每个 seed 独立初始化、独立训练，再对共同 evaluation seeds 测试。

**Standard deviation / 标准差：** 它衡量 episode return 围绕均值的离散程度，不是均值的误差条本身。若要表达均值估计的不确定性，还应报告 standard error 或 confidence interval，并明确样本单位到底是 episode 还是独立 training run。

**Fair comparison / 公平比较：** IQL 和 CQL 必须使用相同训练步数、网络容量说明、evaluation seeds、reward 定义和 action space。只让其中一个算法训练更久，再比较最终回报，没有方法学意义。

## 7. W&B and Weave / 实验追踪与调用追踪

The supplied `spread_iql_wandb_ver.py` and `dqn_wandb_ver.py` are preserved
under `materials/day-03/mpe2-wandb-2026/`. W&B does not change the learning
algorithm; it records the experiment.

Important W&B objects:

- **Project:** collection of related experiments.
- **Run:** one algorithm/seed execution.
- **Config:** fixed hyperparameters and environment settings.
- **History:** metrics indexed by timestep or episode.
- **Artifact:** versioned model or result file connected to a run.

中文对应关系：`project` 像一个实验文件夹；`run` 是其中一次具体训练；`config` 记录这次实验为什么与别次不同；`history` 保存随时间变化的曲线；`artifact` 保存能够复现或继续使用的模型与数据。W&B 的价值不是让模型更聪明，而是防止“跑了很多次，却不知道哪次用了什么参数”。

Useful DQN/MARL metrics include epsilon, replay size, episode return, Q-value
mean, TD error, Huber loss, gradient norm, evaluation mean, and evaluation
standard deviation. For evaluation returns \(R_1,\ldots,R_K\):

\[
\bar R=\frac1K\sum_{k=1}^{K}R_k,
\qquad
\sigma_R=\sqrt{\frac1K\sum_{k=1}^{K}(R_k-\bar R)^2}.
\]

Use one run per `(algorithm, seed)` and hold the training budget, environment,
evaluation seeds, and metric definitions constant. A neat chart cannot repair
an unfair comparison.

### 7.1 How to read the dashboard / 如何看训练曲线

- **Episode return rises / 回报上升：** 可能表示 policy 改善，但训练期包含探索动作，曲线通常比 greedy evaluation 更噪。
- **Loss falls / loss 下降：** 只表示网络更接近当前 TD targets，不自动等于游戏表现提高；targets 自身也在变化。
- **Q-values explode / Q 值爆炸：** 可能是 learning rate 太大、target 不稳定、reward scale 不合适或 bootstrap 错误。
- **TD error stays high / TD error 长期很高：** 网络追不上不断变化的 target，也可能说明 replay 数据分布变化太快。
- **Gradient norm spikes / 梯度尖峰：** 可能预示数值不稳定，可检查 Huber loss、gradient clipping 和输入归一化。
- **Epsilon decreases / epsilon 下降：** 训练从广泛探索过渡到主要利用已学策略；下降过快可能过早锁定差策略。

**Example diagnosis / 诊断例子：** training return 上升、loss 下降，但 evaluation return 不变，可能只是探索期间偶然获得更多奖励，或模型过拟合训练初始布局。反过来，loss 不单调下降但 evaluation 稳步提高也完全可能，因为 RL target 会随着 policy 改善而移动。

**W&B Models** is best for dense scalar histories across training. **W&B
Weave** is best for a small number of structurally rich evaluation traces, such
as the actions, rewards, and termination reason for one episode. API keys must
be stored by `wandb login` or `WANDB_API_KEY`, never in source files.

Models 与 Weave 的区别可以记成：**Models 回答“整体训练趋势怎样”，Weave 回答“这一局具体发生了什么”**。每个训练 step 都做 trace 会产生巨大存储和性能开销，因此训练期记录聚合标量，评估时只追踪少量完整 episode 更合理。

## 8. PyTorch and compact Breakout DQN / PyTorch 与 Breakout DQN 补充

The supplied `message.txt` is a minimal Gymnasium/ALE Breakout DQN. It is now
classified as Day 3 material under
`materials/day-03/dqn-breakout-pytorch-2026/`. It demonstrates replay memory,
epsilon-greedy action selection, a Q-network, Bellman targets, and gradient
descent.

中文理解：DQN 让神经网络近似表格型 Q-learning 的 Q-table。输入是状态，输出是所有离散动作的长期价值估计。训练时只更新实际执行动作对应的那个输出，而不是强迫所有动作同时匹配同一个 target。

例如 batch 中动作 index 是 `[2, 0]`，网络输出形状为 `(2, 4)`，`gather` 会从第一行取第 2 列、第二行取第 0 列，得到这两条 transition 真正需要训练的 \(Q(s_i,a_i)\)。其他列仍通过共享网络参数间接受影响，但不直接与这两个 target 比较。

For this Ubuntu-on-WSL environment, the official PyTorch selector corresponds
to Linux, Pip, Python, and CUDA 12.6. The verified environment is:

```text
Python 3.12.13
PyTorch 2.13.0+cu126
torch.version.cuda 12.6
torch.cuda.is_available() True
NVIDIA GeForce GTX 1650 Ti
```

The compact script uses

\[
y=r+\gamma(1-d)\max_{a'}Q_\theta(s',a')
\]

with the same online network on both sides. Standard DQN stabilizes the target
using a separate network:

\[
y^{DQN}=r+\gamma(1-d)
\max_{a'}Q_{\theta^-}(s',a').
\]

Double DQN separates action selection and evaluation:

\[
y^{Double}=r+\gamma(1-d)
Q_{\theta^-}
\left(s',\arg\max_{a'}Q_\theta(s',a')\right).
\]

为什么需要 target network？如果 online network 同时负责“当前预测”和“训练目标”，一次参数更新会让等式两边一起移动，就像一边追赶一边移动的目标。Target network \(Q_{\theta^-}\) 在一段时间内固定，使 bootstrap target 相对稳定，再定期从 online network 复制参数。

为什么 Double DQN 要再拆一次？普通 DQN 用 `max` 从带噪声的估计中选择最大值，也用同一组估计评价它，容易产生 **overestimation bias（过高估计偏差）**。Double DQN 让 online network 负责选动作，target network 负责给该动作估值，降低选择误差与评价误差的耦合。

The received script flattens a raw \(210\times160\times3\) frame. One float32
frame costs \(210\cdot160\cdot3\cdot4=403{,}200\) bytes, so a 10,000-frame
buffer is already about 3.75 GiB before considering next states and Python
overhead. The repository's maintained Boxing DQN instead uses 84x84 grayscale
frame stacks, uint8 CPU replay, a CNN, target network, epsilon decay, Huber
loss, gradient clipping, checkpoints, and evaluation.

### 8.1 Replay buffer and temporal correlation / 经验回放与时间相关性

连续游戏帧非常相似。如果按产生顺序立刻训练，mini-batch 可能全是同一场景的相邻画面，违背普通随机梯度方法希望样本近似独立的条件。Replay buffer 保存旧 transition 并随机抽样，可以：

1. 打散相邻时间步的相关性；
2. 让一次昂贵的环境交互被多次学习；
3. 混合旧策略和新策略产生的经验。

Replay 也有代价：太小会缺少多样性，太大则包含大量过时策略的数据并消耗内存。在 IQL 中问题更明显，因为旧 transition 还隐含了“当时队友使用的旧 policy”。

### 8.2 Epsilon-greedy / epsilon-greedy 探索

\[
a_t=
\begin{cases}
\text{random action}, & \text{with probability }\epsilon,\\
\arg\max_a Q(s_t,a), & \text{with probability }1-\epsilon.
\end{cases}
\]

若 \(\epsilon=0.1\)，不是说“10% 选择第二好动作”，而是 10% 从整个动作空间随机抽取，包括可能再次抽到当前最优动作。训练早期 Q 值几乎随机，通常使用较大 epsilon；后期逐步降低，但常保留一个小值继续探索。Evaluation 通常设 \(\epsilon=0\)，测量确定性 greedy policy。

### 8.3 Huber loss / Huber 损失

平方误差会把少数非常大的 TD error 平方，早期 Q-value 很不准时可能产生巨大梯度。Huber loss 在误差较小时像 MSE，保留平滑精确更新；误差较大时改为线性增长，降低异常 target 的破坏：

\[
\operatorname{Huber}(\delta)=
\begin{cases}
\frac12\delta^2,& |\delta|\le1,\\
|\delta|-\frac12,& |\delta|>1.
\end{cases}
\]

Installing a CUDA-enabled PyTorch build only makes the GPU available. The model
and each training tensor must explicitly move to `cuda`; replay memory should
normally remain on CPU and only sampled mini-batches should move to the GPU.

GPU 不一定让这个小型 MPE2 网络明显加速。若网络很小、环境 stepping 和 Python 字典操作占大部分时间，CPU 到 GPU 的传输成本可能抵消矩阵计算收益。Atari CNN 与较大 batch 更可能受益。判断是否值得用 GPU 应看 profiler 和 wall-clock time，而不是只看 `torch.cuda.is_available()`。

## 9. Evolution and project connection / 演化脉络与项目联系

The conceptual evolution is:

```text
tabular Q-learning
  -> DQN: neural Q-function + replay + target network
  -> Double/Dueling/PER variants for bias, representation, and sampling

single-agent DQN
  -> IQL: one DQN learner per agent
  -> centralized joint-action Q-learning
  -> scalable MARL methods and CTDE

supervised action prediction
  -> BC
  -> DAgger-style interactive correction
  -> IRL: recover an objective
  -> GAIL: adversarial behavior matching
  -> AIRL: adversarial learning with recoverable reward structure
```

这三条演化路线不是“新算法淘汰旧算法”，而是针对不同失败原因增加结构：

- Q-table 无法覆盖图像等巨大状态空间，所以用神经网络近似，形成 DQN。
- 单一学习者无法直接表示动作组合的协调价值，所以从 IQL 发展到联合价值、value decomposition 和 CTDE。
- BC 只复制见过的动作，遇到 distribution shift 容易崩；IRL、GAIL 和 AIRL 分别转向目标恢复、分布匹配和结构化奖励恢复。

复杂算法只有在解决真实瓶颈时才值得使用。数据错位时换成 GAIL 不会改善；evaluation 不公平时加 W&B 也不会让结论可靠；基础 DQN 尚未稳定时直接加入多智能体与模仿学习，只会让故障来源更多。

For the Boxing project, a conservative learning sequence is:

1. Establish a reproducible reward-trained DQN baseline.
2. Evaluate random and trained policies on fixed unseen seeds.
3. Record demonstrations with episode IDs, timesteps, observations, actions,
   rewards, `terminated`, and `truncated`.
4. Train BC and report held-out action accuracy plus game return and win rate.
5. Add learned-reward methods only after trajectory semantics are reliable.
6. Log algorithm, seed, hyperparameters, training curves, checkpoints, and
   common-seed evaluation in W&B.

Alien pretrained weights cannot be assumed to solve Boxing. The architecture
may be reusable, but visual features and the action head are task-specific and
must be evaluated as transfer learning.

中文理解：Alien 与 Boxing 都是 Atari 图像，但画面物体、动态和有效动作不同。预训练 CNN 可能提供通用边缘或运动特征，也可能已经过度适应 Alien。合理实验应比较“随机初始化”和“加载 Alien encoder”两组，在相同训练预算与 seeds 下看学习速度和最终表现，而不是因为权重能成功加载就宣称 transfer 有效。

## 10. Common misconceptions / 常见误区

1. **“Global reward means every agent learns the same action.” / “共享奖励会让两个 agent 学成一样。”**
   错。它们接收不同 observation、拥有不同 replay 与参数；相同的是优化目标，不是输入和策略。

2. **“CQL always means Conservative Q-Learning.”**
   错。本课程 practical 中 CQL 指 Centralized Q-Learning。阅读论文或代码时必须看上下文和定义。

3. **“A Q-network outputs an action.” / “Q-network 直接输出动作。”**
   更准确地说，它输出每个候选动作的 Q-value，policy 再用 argmax 或 epsilon-greedy 选择动作。

4. **“25 centralized outputs means 25 actions are executed.”**
   错。25 个数对应 25 种动作组合的价值；argmax 选一个组合，再解码成两个环境动作。

5. **“BC does not use rewards, so it has no objective.”**
   错。BC 的 objective 是最小化专家动作的监督分类损失，只是不使用环境 reward。

6. **“IRL discovers the one true reward.”**
   错。奖励通常不可唯一识别，而且候选 reward family 已由研究者的特征或网络结构限制。

7. **“GAIL discriminator accuracy should reach 100%.”**
   不应该把这当目标。若 policy 真能模仿专家，理想判别器应越来越难区分两者；长期 100% 可能表示 policy 没学会或判别器过强。

8. **“Low DQN loss means a strong policy.”**
   错。低 loss 只说明接近当前 bootstrapped targets。必须单独跑无探索或固定探索的 environment evaluation。

9. **“CUDA available means training is using GPU.”**
   错。模型与 sampled tensors 必须明确位于 `cuda`，还要确认计算瓶颈确实适合 GPU。

10. **“A W&B chart makes the experiment reproducible.”**
    不完整。还需要固定代码版本、依赖、seed、环境配置、评估协议和模型 artifact。

## 11. Formula-to-code map / 公式到代码的对应关系

| Mathematical object | Typical code | 中文说明 |
|---|---|---|
| \(o_i\) | `observations[agent]` | 第 \(i\) 个 agent 当前看到的向量 |
| \(Q_i(o_i,\cdot)\) | `q_network(obs_tensor)` | 一次 forward 得到该 agent 所有动作 Q 值 |
| \(\arg\max_aQ(o,a)\) | `q_values.argmax(dim=1)` | 选 Q 值最大的动作 index |
| \(Q(o,a)\) | `q_values.gather(1, actions)` | 从整行输出中取实际执行动作对应的值 |
| \(\max_{a'}Q^-(o',a')\) | `target_q(next_obs).max(1).values` | target network 给下一状态的最佳估值 |
| \((1-d)\) | `1.0 - dones` | episode 结束时关闭下一状态 bootstrap |
| \([o_0;o_1]\) | `np.concatenate([...])` | 拼接 joint observation |
| \(a_0m+a_1\) | `a0 * action_size + a1` | 把动作对编码成 joint index |
| \(\pi_\theta(a\mid s)\) | `softmax(logits)` | BC policy 的动作概率 |
| \(-\log\pi(a_E\mid s)\) | `cross_entropy(logits, labels)` | BC 对专家动作的监督损失 |
| \(D(s,a)\) | `discriminator(features_actions)` | GAIL 判断样本是否像专家 |
| \(w^\top\phi(s,a)\) | `(features * weights).sum(...)` | 线性 IRL reward |
| \(\bar R\) | `np.mean(returns)` | 多个 evaluation episode 的平均回报 |

读代码时应逐项检查 shape。例如 `actions` 给 `gather` 时通常需要 `(B,1)` 的 `long` tensor；BC 的 `cross_entropy` 期望 logits 为 `(B,A)`、class labels 为 `(B,)`。数学公式没写 shape，但程序错误常常就发生在这里。

## 12. Professional glossary / 专业术语详解

| English term | 中文 | Detailed meaning / 详细含义 |
|---|---|---|
| Bootstrap | 自举 | 用当前估计的下一状态价值构造当前 target；不是统计学 bootstrap 抽样 |
| Bellman target | Bellman 目标 | 即时 reward 加折扣后的下一状态估值 |
| TD error | 时序差分误差 | target 与当前 Q prediction 的差 \(\delta=y-Q\) |
| Replay buffer | 经验回放池 | 保存 transition 并随机采样，降低时间相关性 |
| Target network | 目标网络 | 延迟更新的 Q-network 副本，用于稳定 bootstrap target |
| Epsilon-greedy | epsilon 贪心 | 以 \(\epsilon\) 随机探索，否则选最大 Q 动作 |
| Credit assignment | 信用分配 | 判断长期或团队 reward 应归因于哪些动作/agent |
| Non-stationarity | 非平稳性 | 学习过程中数据生成规律变化；MARL 中常由其他 policy 更新导致 |
| Joint observation | 联合观察 | 把多个 agent 的 observation 组合给中央 learner |
| Joint action | 联合动作 | 所有 agent 同一时间步动作的组合 |
| CTDE | 集中训练、分散执行 | 训练可用全局信息，部署时每个 agent 只依赖本地信息 |
| Demonstration | 示范 | 专家产生的有顺序 state-action trajectory |
| Behavioral Cloning | 行为克隆 | 用监督学习直接预测专家动作 |
| Distribution shift | 分布偏移 | 部署时 learner 访问的状态分布不同于专家训练数据 |
| Feature expectation | 特征期望 | policy 轨迹中折扣累计特征的期望 |
| Occupancy measure | 占用分布 | policy 长期访问各 state-action 的频率分布 |
| Proxy reward | 代理奖励 | 代替真实目标的可计算反馈，例如 discriminator 分数 |
| Discriminator | 判别器 | 区分专家样本和 learner 样本的二分类器 |
| Reward shaping | 奖励塑形 | 加入辅助信号改善学习速度，同时尽量保持目标策略不变 |
| Non-identifiability | 不可辨识性 | 多个 reward 都能解释相同专家行为 |
| Logit | 未归一化分数 | softmax/sigmoid 前的神经网络原始输出 |
| One-hot action | 独热动作编码 | 只有选中动作位置为 1 的动作向量 |
| Artifact | 版本化实验产物 | 与实验 run 关联的模型、数据或结果文件 |

## 13. Self-check with answers / 自测题与答案

### Q1. Why can IQL agents choose different actions under the same global reward? / 为什么共享奖励仍会产生不同动作？

Because each learner receives a different observation and has independent parameters. The shared reward aligns the team objective; it does not make their input-conditioned Q-functions identical. / 因为 observation 与网络参数都不同。共享 reward 只统一目标，不统一每个状态下的决策。

### Q2. What are the 25 outputs of centralized Q-learning? / 25 个输出是什么？

They are Q-values for all \(5^2\) pairs of two agents' actions. One argmax selects a pair, and decoding produces the two actions sent to MPE2. / 它们是 25 种动作组合的长期价值，不是 25 个同时执行的动作。

### Q3. Why is BC easier than GAIL? / 为什么 BC 更容易？

BC has fixed inputs and labels and uses ordinary supervised learning. GAIL alternates between a changing policy and changing discriminator, requires environment rollouts, and uses a learned reward. / BC 的数据与标签固定；GAIL 的 policy、判别器和数据分布都在变化。

### Q4. Why is IRL not simply reward-free RL? / IRL 为什么不是“无奖励 RL”？

IRL learns a reward model from demonstrations and then normally solves an RL problem using that learned reward. The reward is inferred rather than manually supplied. / IRL 仍然需要 reward，只是 reward 来自示范推断，而非人工直接给定。

### Q5. Why do we evaluate on common seeds? / 为什么使用相同 evaluation seeds？

Common seeds expose each method to matching initial layouts, reducing environmental variation in pairwise comparisons. Multiple training seeds are still required to measure optimization randomness. / 相同评估种子控制初始局面差异；多个训练种子则衡量训练随机性，两者不能互相替代。

### Q6. When should training stop bootstrapping? / 何时不应 bootstrap？

When the transition truly reaches a terminal state, there is no future return from that episode, so the target is only the immediate reward. Time-limit truncation requires careful treatment depending on the environment semantics. / 自然终止后不存在下一状态回报；时间截断则要按任务定义谨慎处理。

### Q7. What should be the first imitation baseline for Boxing? / Boxing 首个模仿基线应是什么？

BC, after verifying demonstration alignment and preprocessing. It gives the cheapest test of the data pipeline. Environment return and win rate must be reported alongside held-out action accuracy. / 先做 BC 检查数据管线，同时报告环境回报和胜率，不能只看动作准确率。

## 14. Source map and status / 资料地图与状态

| Material | Location | Status |
|---|---|---|
| Live Day 3 photographs | summarized in `imitation-learning.md` | Read and integrated |
| 2024 and 2026 imitation slides | `materials/` and `materials/day-03/imitation-learning-2026/` | Read and integrated; 2026 takes priority |
| BC/IRL/GAIL Python and BC weights | `materials/day-03/imitation-learning-2026/source/` | Preserved; syntax/shape inspection completed |
| Missing expert demonstrations | documented beside the source | Not supplied; full training unavailable |
| MPE2 tutorial and official links | explained in `mpe2-practical.md` | Followed and locally evaluated |
| W&B IQL/DQN versions | `materials/day-03/mpe2-wandb-2026/source/` | Preserved and analyzed |
| Breakout DQN `message.txt` | `materials/day-03/dqn-breakout-pytorch-2026/source/` | Preserved; forward/update smoke tests passed |
| PyTorch official setup | linked in `dqn-breakout-pytorch.md` | CUDA 12.6 environment verified |

All files under `notes/` and `materials/` are local course material and remain
excluded by `.gitignore`. No note, slide, photograph, received teaching script,
model weight, or W&B-generated run is intended for the GitHub code repository.
