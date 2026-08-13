# Day 3 Practical - MPE2 Simple Spread / MPE2 多智能体实践

> Day 3 master guide / Day 3 总讲解：`README.md`

## 中文精读导读 / Detailed bilingual reading guide

这份 practical 研究同一个 cooperative task 的三种控制方式：随机策略用于确认环境和指标能运行；IQL 让每个 agent 独立学习；集中式 Q-learning 用一个网络直接评价动作组合。三者必须在相同 reward、episode 长度和 evaluation seeds 下比较。

### 先把环境看成一张数据流图 / Environment as a data flow

```text
两个局部 observations
  -> 两个独立动作，或一个 joint-action index
  -> MPE2 同时执行动作
  -> 两个 next observations
  -> 每个 agent 收到同一个 global reward
  -> transition 进入 replay buffer
```

`local_ratio=0.0` 表示完全使用团队奖励。奖励越接近 0 通常越好，因为它主要是负的 landmark 距离。两个 agent 得到同一个 reward，不应把两份相同数值相加后再报告，否则会把一个团队信号重复计算。

### IQL 的两个 observation 与两个网络 / Two observations, two networks

IQL 并不是把 `o_0` 和 `o_1` 一起输入同一个模型。它计算

\[
Q_0(o_0,\cdot)\in\mathbb R^5,
\qquad
Q_1(o_1,\cdot)\in\mathbb R^5.
\]

所以一次决策总共产生 10 个 Q 值，分别做两次 argmax，得到两个动作。每个 learner 有自己的 online network、target network、optimizer 和 replay buffer。共享 reward 负责让目标一致，但协调只能通过长期样本间接学到。

### 集中式网络为什么有 25 个输出？ / Why 25 centralized outputs?

两个 agent 各有 5 个动作，一共存在 \(5\times5=25\) 个动作组合。集中式网络输入 `[o_0;o_1]`，输出每一种 `(a_0,a_1)` 的 joint Q-value。只做一次 argmax 得到 joint index，再解码成两个动作。它能直接比较“两个都往左”和“一个往左、一个往右”的团队效果，但输出宽度随 agent 数量按 \(m^N\) 指数增长。

### 结果怎么解释 / How to interpret results

本机 100 个共同 evaluation seeds 上，Random、IQL、集中式 Q-learning 的平均团队回报分别为 `-40.361`、`-17.236`、`-16.745`。两个学习方法都明显优于 random；集中式均值只比 IQL 高约 `0.491`，同时标准差更大。这个结果支持“本次运行中两者都学会了，CQL 均值略高”，不支持“CQL 普遍优于 IQL”。

### W&B 记录什么 / What W&B should record

每个 `(algorithm, training_seed)` 建立一个 run。`config` 保存学习率、gamma、buffer、batch、epsilon schedule、环境参数和训练预算；`history` 保存 return、loss、TD error、Q mean、gradient norm 和 greedy evaluation；`artifact` 保存模型。W&B 负责证据链，不负责替代公平实验设计。

> Environment source / 环境来源: [Farama Foundation/MPE2](https://github.com/Farama-Foundation/MPE2)
>
> Documentation / 官方文档: [mpe2.farama.org](https://mpe2.farama.org/)
>
> Course tutorial / 课程教程: [Hautamaki-lab/Summer-School-2026](https://github.com/Hautamaki-lab/Summer-School-2026), `MARL-IQL-CQL-Tutorial.md`

## 1. Practical roadmap / 实践路线

```text
Simple Spread random baseline
    -> Independent Q-Learning (IQL)
    -> Centralized Q-Learning (CQL)
    -> Random vs IQL vs CQL evaluation
```

注意：这里的 **CQL** 是 **Centralized Q-Learning（集中式 Q 学习）**，不是 offline RL 中的 **Conservative Q-Learning（保守 Q 学习）**。

### 1.1 Formula notation / 公式符号表

| Symbol | Meaning / 含义 |
|---|---|
| \(N\) | number of agents / 智能体数量；本练习 \(N=2\) |
| \(L=|\mathcal L|\) | number of landmarks / 地标数量；本练习 \(L=2\) |
| \(m=|\mathcal A_i|\) | actions available to each agent / 每个智能体的动作数；本练习 \(m=5\) |
| \(i\in\{0,\ldots,N-1\}\) | agent index / 智能体编号 |
| \(\ell\in\mathcal L\) | landmark index / 地标编号 |
| \(o_i,o_i'\) | current and next observation of agent \(i\) / 当前与下一局部观测 |
| \(a_i,a_i'\) | current and candidate next action of agent \(i\) / 当前与候选下一动作 |
| \(p_i,p_\ell\) | position vectors of agent \(i\) and landmark \(\ell\) / 位置向量 |
| \(r_{global}\) | shared team reward / 所有智能体共享的团队奖励 |
| \(r_{local,i}\) | local reward component for agent \(i\) / 智能体 \(i\) 的局部奖励 |
| \(\lambda\) | local/global mixing coefficient; `local_ratio` / 奖励混合系数 |
| \(\gamma\in[0,1)\) | discount factor / 折扣因子；越大越重视长期回报 |
| \(d_i\in\{0,1\}\) | done flag / 终止标志；结束时为 1，否则为 0 |
| \(Q_i,Q_{central}\) | individual and centralized action-value functions / 个体与集中式动作价值函数 |
| \(y_i,y\) | fixed TD learning targets / TD 训练目标，不是网络当前预测 |

Unless stated otherwise, \(t\) denotes an environment time step. A prime, as in \(o_i'\), means the value after one environment step. / 除非另外说明，\(t\) 表示环境时间步；右上角的撇号表示执行一次联合动作之后的下一个值。

## 2. MPE2 / Multi-Particle Environments 2

MPE2 is a collection of communication-oriented multi-agent environments. Particle agents may move, observe one another, communicate in some tasks, interact with landmarks, and push other particles.

The practical uses the current package directly:

```python
from mpe2 import simple_spread_v3
```

Install and verify:

```bash
conda activate pettingzoo
python -m pip install mpe2
python -c "from mpe2 import simple_spread_v3; print('MPE2 loaded successfully')"
```

Do not use the deprecated old import path `pettingzoo.mpe` for this tutorial.

## 3. Simple Spread / Simple Spread 环境

The tutorial configuration has two agents and two landmarks:

```python
env = simple_spread_v3.parallel_env(
    N=2,
    local_ratio=0.0,
    max_cycles=25,
    continuous_actions=False,
)
```

| Setting | Meaning / 含义 |
|---|---|
| `N=2` | Two agents and two landmarks |
| `local_ratio=0.0` | Fully shared global team reward |
| `max_cycles=25` | At most 25 simultaneous environment steps |
| `continuous_actions=False` | Five discrete movement actions per agent |

Actions:

```text
0: no action
1-4: movement directions
```

The observation is a numeric vector, not an image. It contains information such as the agent's velocity and position, relative landmark positions, and relative positions of other agents.

## 4. Parallel PettingZoo API / 并行 API

All active agents act in the same environment step:

```python
actions = {
    "agent_0": action_0,
    "agent_1": action_1,
}

observations, rewards, terminations, truncations, infos = env.step(actions)
```

- **Parallel API（并行 API）**: submit all agents' actions together.
- **Joint action（联合动作）**: the tuple of all individual actions at one time step.
- **Termination（终止）**: the task ended because of an environment terminal condition.
- **Truncation（截断）**: the episode ended because of a limit such as `max_cycles`.

## 5. Reward interpretation / 奖励解释

Simple Spread rewards agents for covering landmarks efficiently. The reward is distance-based and normally negative:

```text
-10 is better than -30
0 is the ideal direction
```

With `local_ratio=0.0`, both agents receive the same team reward. The code can therefore accumulate `rewards["agent_0"]` once as the team reward; adding both agents' identical rewards would double-count the same team signal.

### 5.1 Global reward / 全局奖励

A **global reward（全局奖励）** evaluates the joint team outcome and is shared by every agent.

In Simple Spread, for each landmark, the environment finds the closest agent. The global coverage reward is the negative sum of those closest-agent distances:

\[
r_{global}
= -\sum_{\ell\in\mathcal L}
\min_{i\in\{1,\ldots,N\}}
\lVert p_i-p_\ell\rVert_2.
\]

- \(\mathcal L\): set of landmarks / 地标集合。
- \(p_i\): position of agent \(i\).
- \(p_\ell\): position of landmark \(\ell\).
- The minus sign means shorter total distance gives a higher, less-negative reward.

Read the formula from the inside out / 从内向外读公式：

1. \(\lVert p_i-p_\ell\rVert_2\) computes the Euclidean distance from agent \(i\) to landmark \(\ell\).
2. \(\min_i\) keeps only the distance of the closest agent to that landmark.
3. \(\sum_\ell\) adds the closest-agent distance for every landmark.
4. The leading minus sign turns smaller coverage distance into larger reward.

Numeric example / 数值例子：if the nearest-agent distances for the two landmarks are \(0.2\) and \(0.5\), then

\[
r_{global}=-(0.2+0.5)=-0.7.
\]

If the agents improve coverage to distances \(0.1\) and \(0.2\), the reward becomes \(-0.3\). Since \(-0.3>-0.7\), the second arrangement is better.

The `min` is important: a landmark is considered well covered when **at least one** agent is near it. If both agents crowd around the same landmark, the uncovered landmark remains far from every agent and keeps the team reward poor.

This reward therefore creates cooperation without explicitly assigning “agent 0 must cover landmark 0”. The division of labor must emerge from learning.

### 5.2 Local reward / 局部奖励

The current MPE2 documentation describes the local component as an individual collision penalty: an agent is penalized when it collides with another agent.

The effective per-agent reward is conceptually mixed as:

\[
r_i
= (1-\lambda)r_{global}
+ \lambda r_{local,i},
\]

where \(\lambda=\texttt{local\_ratio}\).

| `local_ratio` | Interpretation / 含义 |
|---:|---|
| `0.0` | Pure global team reward; all agents receive the same reward |
| `0.5` | Equal weighting of global coverage and local collision signal |
| `1.0` | Pure local component; global coverage is ignored |

The course tutorial sets `local_ratio=0.0`, so:

\[
r_0=r_1=r_{global}.
\]

Numeric example / 数值例子：suppose \(r_{global}=-2\), \(r_{local,i}=-1\), and \(\lambda=0.25\). Then

\[
r_i=(1-0.25)(-2)+0.25(-1)=-1.75.
\]

With the tutorial setting \(\lambda=0\), this simplifies to \(r_i=r_{global}=-2\); the local term contributes nothing.

### 5.3 Why global reward helps and hurts / 全局奖励的优缺点

**Advantage - incentive alignment（激励一致）:** both agents optimize the same team objective, so one agent does not benefit by harming team coverage.

**Difficulty - multi-agent credit assignment（多智能体信用分配）:** when the global reward improves, an individual agent cannot directly tell whether the improvement came from its own action or its teammate's action.

Example:

```text
agent_0 moves toward an uncovered landmark
agent_1 simultaneously moves away from another landmark
team reward changes only once
```

Both agents observe the same final reward even though their contributions differ. Shared reward says **whether the team improved**, not **who deserves credit**.

## 6. Random baseline / 随机基线

The random policy ignores observations:

```python
action_0 = env.action_space("agent_0").sample()
action_1 = env.action_space("agent_1").sample()
```

Why keep a random baseline?

- verifies that the environment runs
- provides a minimum performance reference
- catches evaluations where a learned policy is not actually improving

## 7. Shared DQN components / 共用 DQN 组件

`dqn.py` contains:

| Component | Purpose / 作用 |
|---|---|
| `QNetwork` | Predict one Q-value per available action |
| `ReplayBuffer` | Store and randomly sample transitions |
| `train_dqn()` | Compute TD targets and update the network |
| `update_target()` | Copy online-network weights to target network |

DQN target:

\[
y = r + \gamma(1-d)\max_{a'}Q_{\text{target}}(s',a').
\]

This is a one-step **temporal-difference target (TD target，时序差分目标)**:

- \(r\): reward observed immediately after the current action.
- \(\max_{a'}Q_{target}(s',a')\): estimated best future return from the next state.
- \(1-d\): removes the future term when the episode has ended.
- \(y\): target value used in a loss such as \((Q_{online}(s,a)-y)^2\).

Numeric example / 数值例子：if \(r=-0.4\), \(\gamma=0.95\), \(d=0\), and the largest target-network value at the next state is \(1.2\), then

\[
y=-0.4+0.95(1-0)(1.2)=0.74.
\]

If the transition terminates the episode, \(d=1\), so \(y=-0.4\); no future value may be collected after termination.

The tutorial actually uses PyTorch smooth L1 loss, equivalent to the Huber loss with threshold 1. Let the TD error be

\[
\delta=Q_{online}(s,a)-y.
\]

Then

\[
\mathcal L_{Huber}(\delta)=
\begin{cases}
\frac{1}{2}\delta^2, & |\delta|<1,\\
|\delta|-\frac{1}{2}, & |\delta|\ge 1.
\end{cases}
\]

Small errors are treated quadratically for smooth optimization; large errors grow linearly, making training less sensitive to outliers than pure squared error.

The target network changes less frequently than the online Q-network, making the bootstrap target more stable.

### 7.1 Epsilon-greedy exploration / epsilon-greedy 探索

At training step \(t\), the tutorial linearly decreases exploration from \(\epsilon_{start}=1.0\) to \(\epsilon_{end}=0.05\) over \(T_{decay}=10{,}000\) steps:

\[
\epsilon_t
=\max\left(
\epsilon_{end},
\epsilon_{start}
-(\epsilon_{start}-\epsilon_{end})\frac{t}{T_{decay}}
\right).
\]

The action rule is

\[
a_t=
\begin{cases}
\text{uniform random action}, & \text{with probability }\epsilon_t,\\
\arg\max_a Q(o_t,a), & \text{with probability }1-\epsilon_t.
\end{cases}
\]

Thus \(\epsilon_0=1.0\), \(\epsilon_{5000}=0.525\), and every step at or after \(10{,}000\) uses \(\epsilon_t=0.05\). The remaining 5% exploration prevents the behavior from becoming completely deterministic during training.

## 8. Independent Q-Learning / 独立 Q 学习

IQL gives each agent a separate learner:

```text
agent_0 observation -> Q_0 -> action_0
agent_1 observation -> Q_1 -> action_1
```

Each agent has its own:

- Q-network
- target network
- optimizer
- replay buffer

The agents cooperate because their rewards are shared, not because their neural-network parameters are shared.

Formally, each agent \(i\) learns its own action-value function:

\[
Q_i(o_i,a_i),
\]

Conceptually, a Q-value is the expected discounted return after choosing \(a_i\) under observation \(o_i\) and then following the policy:

\[
Q_i(o_i,a_i)
=\mathbb E\!\left[
\sum_{k=0}^{\infty}\gamma^k r_{t+k}
\;\middle|\;
o_{i,t}=o_i,\ a_{i,t}=a_i
\right].
\]

Here \(k\) counts how many steps into the future the reward occurs. In a partially observed multi-agent environment, \(o_i\) may not uniquely identify the full world state, so a neural network can only approximate this conditional expectation from its available information.

where \(o_i\) is that agent's observation and \(a_i\) is its individual action. Its DQN target in this exercise is:

\[
y_i
= r_{global}
+ \gamma(1-d_i)
\max_{a_i'}Q_{i,target}(o_i',a_i').
\]

The maximization is only over agent \(i\)'s five next actions. It does not directly optimize the teammate's next action. The shared reward links the agents' objectives, while their Q-functions and maximizations remain separate.

Even though both learners receive the same \(r_{global}\), they store separate transitions:

```text
agent_0 buffer: (o_0, a_0, r_global, o'_0, done_0)
agent_1 buffer: (o_1, a_1, r_global, o'_1, done_1)
```

This gives the central distinction:

```text
independent learning architecture
does not imply
independent or selfish rewards
```

In this practical:

- learning is independent: separate networks, buffers, and optimizers
- the objective is cooperative: the same global reward goes to both agents
- execution is decentralized: each IQL agent selects an action from its own observation

### 8.1 How IQL works step by step / IQL 逐步流程

1. Each agent receives its own observation.
2. Each agent independently uses epsilon-greedy action selection.
3. The action dictionary forms one joint action.
4. The environment advances using both actions together.
5. Both agents receive the global team reward.
6. Each agent stores its own local transition with that shared reward.
7. Each agent samples and updates its own DQN independently.
8. The process repeats while both policies change.

### 8.2 Why the environment becomes non-stationary / 为什么出现非平稳性

Single-agent Q-learning assumes the transition/reward behavior is sufficiently stable while learning. For agent 0 here:

\[
P(o_0'\mid o_0,a_0)
\]

also depends on agent 1's action and policy. While agent 1 learns, its behavior changes, so the effective dynamics experienced by agent 0 change too:

\[
P(o_0'\mid o_0,a_0,\pi_1^{(t)})
\ne
P(o_0'\mid o_0,a_0,\pi_1^{(t+1)}).
\]

In this formula, superscript \(t\) denotes a **learning iteration**, not an environment time step: \(\pi_1^{(t)}\) is agent 1's policy before another update and \(\pi_1^{(t+1)}\) is its changed policy afterward.

From agent 0's perspective, similar observation-action pairs can lead to different outcomes as the teammate evolves. This violates the stationary-environment intuition behind ordinary DQN and can make replay-buffer data stale.

### 8.3 Why IQL can still work here / 为什么这个练习中 IQL 仍可能有效

- only two agents
- only five actions per agent
- short 25-step episodes
- fully shared cooperative reward
- relatively simple continuous state geometry
- both agents use similar learning schedules

These conditions make IQL a useful baseline, even though it has no formal guarantee of stable convergence in general multi-agent learning.

Main benefit: the individual action space stays small.

Main limitation: from one agent's perspective, the environment becomes **non-stationary（非平稳）** while the other agent's policy is learning and changing.

## 9. Centralized Q-Learning / 集中式 Q 学习

CQL concatenates both observations into one joint state and chooses an action pair with one network:

\[
Q(s_0,s_1,a_0,a_1).
\]

### 9.1 Different observations in IQL / IQL 中不同 observation 如何进入 Q-network

At one environment step, the two agents receive different observation vectors:

\[
o_0 \ne o_1.
\]

Even when both contain information about the same world, they are agent-centered:

- `agent_0`'s self-position is agent 0's position
- `agent_1`'s self-position is agent 1's position
- relative positions are measured from different reference points
- the “other agent” field refers to a different entity

IQL sends them to different networks:

```text
o_0 -> Q-network 0 -> [Q_0(o_0, 0), ..., Q_0(o_0, 4)] -> a_0
o_1 -> Q-network 1 -> [Q_1(o_1, 0), ..., Q_1(o_1, 4)] -> a_1
```

Each network outputs five Q-values, one per individual action. During exploitation:

\[
a_i=\arg\max_{a\in\{0,\ldots,4\}}Q_i(o_i,a).
\]

The `action()` method implements this:

```python
q_values = q_network(observation)
action = q_values.argmax(dim=1)
```

The Q-network does not directly output the physical movement vector. It outputs action values; `argmax` returns the discrete action index with the highest estimated return.

### 9.2 Joint observation in CQL / CQL 中的联合 observation

CQL combines the two different observations:

\[
o_{joint}=[o_0;o_1],
\]

where the semicolon means concatenation. In code:

```python
joint_observation = np.concatenate(
    [observations["agent_0"], observations["agent_1"]]
)
```

The centralized Q-network therefore has access to both agents' views at once:

```text
[o_0, o_1]
    -> one centralized Q-network
    -> 25 Q-values, one for each (a_0, a_1) pair
    -> one joint-action index
    -> decode to a_0 and a_1
```

Formally:

\[
a_{joint}
= \arg\max_{(a_0,a_1)\in\mathcal A_0\times\mathcal A_1}
Q_{central}(o_0,o_1,a_0,a_1).
\]

This is different from asking two independent Q-networks for two separate argmax actions. The centralized network can assign a value to the **combination** itself.

Example:

```text
(agent_0 moves left, agent_1 moves right)  -> high Q-value
(agent_0 moves left, agent_1 moves left)   -> lower Q-value
```

The first pair may cover two landmarks; the second may make both agents crowd the same location. An independent network has no direct joint-action output for expressing this comparison.

With five actions per agent:

\[
|\mathcal A_{joint}|=5\times5=25.
\]

In general, if action-space sizes may differ between agents,

\[
|\mathcal A_{joint}|
=\prod_{i=0}^{N-1}|\mathcal A_i|.
\]

If every agent has the same \(m\) actions, the product reduces to \(m^N\).

Joint-action encoding:

```python
joint_action = action_0 * action_size_1 + action_1
```

Decoding:

```python
action_0 = joint_action // action_size_1
action_1 = joint_action % action_size_1
```

For example, if each agent has five actions and the centralized network selects joint index 17:

```text
action_0 = 17 // 5 = 3
action_1 = 17 % 5  = 2
```

So joint index 17 represents action pair `(3, 2)`.

### 9.3 CQL replay transition / CQL 的 replay transition

IQL stores two transitions per environment step, one per agent. CQL stores one centralized transition:

\[
(o_{joint},a_{joint},r_{global},o'_{joint},done).
\]

Its DQN target is:

\[
y
= r_{global}
+ \gamma(1-d)
\max_{a'_{joint}}
Q_{central,target}(o'_{joint},a'_{joint}).
\]

The network learns which joint action is valuable for the shared team objective.

### 9.4 Information and execution / 信息与执行方式

The tutorial's IQL policies can execute separately because each needs only its own observation. The CQL policy needs both observations before either action can be selected.

```text
IQL: decentralized action selection
CQL: centralized action selection
```

If communication with a central controller were unavailable at deployment, this CQL policy could not run as written. That is why it is centralized execution rather than CTDE.

Main benefit: one learner reasons directly about coordination and action combinations.

Main limitation: joint-action size grows exponentially. With \(n\) agents and \(m\) actions each, there are \(m^n\) joint actions.

The joint observation grows additively instead. If agent \(i\)'s observation has dimension \(d_i\), then

\[
d_{joint}=\sum_{i=0}^{N-1}d_i.
\]

Thus, concatenating observations increases the input width linearly, while enumerating every joint action increases the output width multiplicatively.

## 10. IQL versus CQL / IQL 与 CQL 对比

| Property | IQL | CQL |
|---|---|---|
| Number of learned Q-functions | One per agent | One centralized Q-function |
| Input | Individual observation | Concatenated joint observation |
| Output | Individual action values | Joint-action values |
| Coordination information | Indirect through reward | Explicit in joint Q-values |
| Scaling | Better action-space scaling | Exponential joint-action growth |
| Non-stationarity | Other learners change | Reduced inside centralized learner |

### 10.1 Exactly how many outputs? / 最后到底有多少个输出？

For this practical:

```text
number of agents N = 2
actions per agent m = 5
```

| Method | Network objects | Q-values produced per decision | Selected result | Actions sent to environment |
|---|---:|---:|---|---:|
| Random | 0 | 0 | Two sampled actions | 2 |
| IQL | 2 online Q-networks | `2 x 5 = 10` values across both networks | One argmax per network | 2 |
| CQL | 1 online Q-network | `5 x 5 = 25` joint-action values | One joint argmax, decoded | 2 |

Important interpretation:

- IQL's ten Q-values are split into **two separate groups of five**. They do not form one ten-action space.
- CQL's 25 Q-values form **one joint-action space** containing every pair `(a_0, a_1)`.
- Both methods ultimately send exactly one action per agent to `env.step(actions)`.

General case with \(N\) agents and \(m\) actions per agent:

\[
\text{IQL total Q-values evaluated per step}=N\times m,
\]

\[
\text{centralized joint-action Q-values}=m^N.
\]

Example with 3 agents and 5 actions each:

```text
IQL: 3 x 5 = 15 Q-values across three networks
CQL: 5^3 = 125 joint-action Q-values in one network
final environment action dictionary: still 3 actions
```

This exponential \(m^N\) growth is the **combinatorial explosion（组合爆炸）** that limits centralized joint-action DQN.

This exercise uses centralized control during both training and evaluation. It is not yet **CTDE (Centralized Training with Decentralized Execution，集中训练、分散执行)** because the CQL policy still needs both agents' observations to choose actions at execution time.

## 11. Evaluation / 评估

The evaluation uses 100 unseen games per method and the same seeds for all methods.

One episode's undiscounted team return is the sum of its step rewards:

\[
R_k=\sum_{t=0}^{T_k-1}r_{global,t}^{(k)},
\]

where \(T_k\le25\) is the number of steps in evaluation episode \(k\). The code adds only the reward entry for agent 0 because both agents receive the same \(r_{global,t}\); summing both entries would count the same team reward twice.

For \(K=100\) evaluation episodes with team returns \(R_1,\ldots,R_K\), the reported mean is

\[
\bar R=\frac{1}{K}\sum_{k=1}^{K}R_k,
\]

and the population standard deviation used to describe run-to-run variability is

\[
\sigma_R
=\sqrt{\frac{1}{K}\sum_{k=1}^{K}(R_k-\bar R)^2}.
\]

Because Simple Spread returns are usually negative, a larger mean is better: for example, \(-12>-20\). Improvement over random is

\[
\Delta_{method-random}=\bar R_{method}-\bar R_{random},
\]

and the centralized-versus-independent difference is

\[
\Delta_{CQL-IQL}=\bar R_{CQL}-\bar R_{IQL}.
\]

A positive \(\Delta\) means the method named first achieved the higher average reward.

Report:

- mean team reward / 平均团队奖励
- standard deviation / 标准差
- improvement over random / 相对随机策略的提升
- CQL minus IQL / CQL 与 IQL 的差值

Using the same seeds creates a paired comparison: each method faces identical initial layouts.

Do not conclude that CQL is universally better from one small run. Reliable conclusions are limited to this environment, configuration, seed set, and training budget.

### 11.1 Actual run on this computer / 本机实测结果

Both learned methods were trained for 50,000 environment steps with seed 42. The last printed 100-game training means were:

| Training method | Last printed 100-game mean |
|---|---:|
| IQL | -18.284 |
| CQL | -18.800 |

The official tutorial evaluation script then tested each method on the same 100 unseen seeds:

| Method | Mean team return \(\bar R\) | Standard deviation \(\sigma_R\) | Improvement over random |
|---|---:|---:|---:|
| Random | -40.361 | 12.750 | 0.000 |
| IQL | -17.236 | 6.291 | 23.125 |
| CQL | -16.745 | 8.674 | 23.615 |

Substitution into the comparison formulas gives

\[
\Delta_{IQL-random}
=-17.236-(-40.361)
\approx23.125,
\]

\[
\Delta_{CQL-random}
=-16.745-(-40.361)
\approx23.615,
\]

and

\[
\Delta_{CQL-IQL}
=-16.745-(-17.236)
\approx0.491.
\]

The script computes differences from unrounded episode returns, so the displayed subtraction may differ by \(0.001\) from arithmetic using the three-decimal table.

Interpretation / 结果解释：

- Both learned policies substantially outperformed random action selection.
- CQL had the best mean in this run, but its advantage over IQL was only \(0.491\).
- CQL also had the larger standard deviation (\(8.674\) versus \(6.291\)), so its evaluation performance varied more across starting layouts.
- One seed and one training run are not enough to claim that CQL is generally superior. A stronger experiment would repeat training with multiple seeds and compare confidence intervals.

## 12. Local environment / 本机环境

Verified configuration:

```text
Ubuntu via WSL
Python 3.12.13
PyTorch 2.13.0+cu126
CUDA runtime 12.6
GPU NVIDIA GeForce GTX 1650 Ti
MPE2 1.1.0
```

The networks are small, so environment stepping and Python overhead may dominate; GPU availability does not guarantee a dramatic speedup.

## 13. Files and commands / 文件与命令

```text
~/pettingzoo-marl/
├── README.md
├── spread_random.py
├── dqn.py
├── spread_iql.py
├── spread_cql.py
├── spread_evaluate.py
└── checkpoints/
```

```bash
conda activate pettingzoo
cd ~/pettingzoo-marl

python spread_random.py
python spread_iql.py
python spread_cql.py
python spread_evaluate.py
```

## 14. Self-check / 自测题

1. Why does the environment receive an action dictionary instead of one action?
2. Why is a reward closer to zero better in Simple Spread?
3. Why should the shared reward not be summed over both agents?
4. What exactly makes IQL “independent”?
5. Why is another learning agent a source of non-stationarity?
6. How are two individual actions encoded into one joint-action index?
7. Why does centralized tabular or DQN control scale poorly with more agents?
8. Is the tutorial CQL implementation CTDE? Why or why not?

## 15. Weights & Biases experiment tracking / W&B 实验追踪

The optional `wandb_ver.py` adds **Weights & Biases (W&B，实验追踪平台)** to the practical without changing the original teacher-provided scripts. W&B records a training run's hyperparameters, metric history, system utilization, evaluation results, and model files in one comparable experiment page.

Official references / 官方入口：

- [Weights & Biases](https://wandb.ai/)
- [W&B experiment logging](https://docs.wandb.ai/models/track/log)
- [W&B Artifacts](https://docs.wandb.ai/models/artifacts)

### 15.1 Run, config, metric, and artifact / 四个基本对象

- **Run（一次实验运行）**: one execution of `wandb_ver.py`, such as IQL with seed 42.
- **Config（实验配置）**: fixed hyperparameters such as learning rate, discount factor, seed, and training steps.
- **Metric（指标）**: a value that changes during or after training, such as loss or episode return.
- **Artifact（版本化产物）**: a saved output such as a `.pth` model checkpoint, connected to the run that produced it.

W&B is an observation and bookkeeping layer:

\[
\text{environment + algorithm}
\longrightarrow
\text{metrics and checkpoints}
\longrightarrow
\text{W&B run}.
\]

It does **not** alter the reward, Q-learning target, neural network, or selected actions. If two executions use identical seeds and software behavior, enabling logging should not intentionally change the learning algorithm.

### 15.2 Logged training metrics / 训练指标

| W&B key | Mathematical meaning / 数学含义 |
|---|---|
| `global_step` | current environment step (t) |
| `train_episode` | number of completed episodes |
| `train_episode_return` | (R_k=\sum_{t=0}^{T_k-1}r_t) for episode (k) |
| `train_mean_return_100` | mean of the most recent at most 100 episode returns |
| `train_epsilon` | current exploration probability \(\epsilon_t\) |
| `train_loss` | centralized DQN Huber loss |
| `train_loss_agent_0`, `train_loss_agent_1` | separate IQL learner losses |
| `train_loss_mean` | arithmetic mean of the two available IQL losses |
| `train_replay_size` | current number of stored replay transitions |

For the rolling window \(W_k\), containing the latest \(\min(k,100)\) completed episodes, the plotted mean is

\[
\bar R_{100,k}
=\frac{1}{|W_k|}
\sum_{j\in W_k}R_j.
\]

This smooths noisy individual returns. It is still a training diagnostic, not the final evaluation score, because training episodes include epsilon-greedy exploration.

IQL logs two losses because it has two independent Q-networks:

\[
\mathcal L_0
=\mathcal L_{Huber}(Q_0(o_0,a_0)-y_0),
\qquad
\mathcal L_1
=\mathcal L_{Huber}(Q_1(o_1,a_1)-y_1).
\]

Their dashboard mean is

\[
\mathcal L_{mean}=\frac{\mathcal L_0+\mathcal L_1}{2}.
\]

A lower TD loss means the network better fits its current bootstrap targets. It does not by itself prove that the policy earns a better team return; reward curves and evaluation must also be checked.

### 15.3 Evaluation metrics / 评估指标

After training, epsilon is removed and the greedy policy is tested on unseen seeds. For (K) evaluation episodes:

\[
\bar R_{eval}
=\frac{1}{K}\sum_{k=1}^{K}R_k,
\qquad
\sigma_{eval}
=\sqrt{\frac{1}{K}\sum_{k=1}^{K}(R_k-\bar R_{eval})^2}.
\]

These are logged as `evaluation_mean_return` and `evaluation_std_return`. The individual returns are also stored in a W&B Table so unusually easy or difficult starting layouts can be inspected instead of hiding everything inside one average.

### 15.4 Online, offline, and disabled modes / 三种运行模式

```bash
# Upload metrics live; requires wandb login and internet.
python wandb_ver.py --algorithm iql --wandb-mode online

# Save a complete local W&B run and upload it later.
python wandb_ver.py --algorithm cql --wandb-mode offline
wandb sync wandb/offline-run-*

# Run the same training code without creating a W&B record.
python wandb_ver.py --algorithm iql --wandb-mode disabled
```

Never put a W&B API key in source code or commit it to Git. Use `wandb login` or the `WANDB_API_KEY` environment variable. The local `wandb/` directory is generated experiment data and is ignored by this repository.

### 15.5 Comparing runs professionally / 如何专业比较实验

Use one W&B run per pair \((\text{algorithm},\text{seed})\). For a fair IQL/CQL comparison, hold constant:

- total environment steps
- environment configuration and evaluation seeds
- learning rate, gamma, replay size, and batch size
- epsilon schedule and target-update frequency

Then repeat with several seeds. If run (s) produces evaluation mean \(\bar R_s\), the across-seed mean is

\[
\bar R_{seeds}=\frac{1}{S}\sum_{s=1}^{S}\bar R_s.
\]

This separates a repeatable algorithm trend from one lucky initialization. A single W&B chart makes experiments easier to inspect, but it does not replace multiple seeds or careful experimental design.

## 16. W&B Weave traces / Weave 调用追踪

The supplied Weave Quickstart demonstrates tracing language-model calls, but **Weave is not limited to LLMs**. Decorating an ordinary Python function with `@weave.op` records its inputs, outputs, runtime, code version, and nested calls. In this practical, `weave_ver.py` applies the same mechanism to trained RL policy evaluation.

Reference / 参考: [W&B Weave - Track your own operations](https://weave-docs.wandb.ai/guides/integrations/groq/#track-your-own-ops)

### 16.1 W&B Models versus Weave / 两种工具的分工

```text
wandb_ver.py
    -> training curves, hyperparameters, system metrics, comparisons, artifacts

weave_ver.py
    -> evaluation call tree, per-episode inputs/outputs, step trajectory debugging
```

- **W&B run history** answers: “Did reward improve over 50,000 steps?”
- **Weave trace** answers: “What exactly happened in evaluation seed 10003?”

They are complementary. A metric chart summarizes many events; a trace preserves the structure of a particular computation.

### 16.2 Trace hierarchy / Trace 层级

The code creates a nested call tree:

```text
evaluate_policy(algorithm, episodes, first_seed)
|-- evaluate_episode(seed=10000)
|-- evaluate_episode(seed=10001)
`-- evaluate_episode(seed=10002)
```

For evaluation episode (k), the child trace records a trajectory

\[
\tau_k
=\left(
o_{0,t},o_{1,t},a_{0,t},a_{1,t},r_t
\right)_{t=0}^{T_k-1},
\]

and computes its team return

\[
R_k=\sum_{t=0}^{T_k-1}r_t.
\]

The parent trace aggregates the child outputs:

\[
\bar R=\frac{1}{K}\sum_{k=1}^{K}R_k,
\qquad
\sigma_R
=\sqrt{\frac{1}{K}\sum_{k=1}^{K}(R_k-\bar R)^2}.
\]

This nesting is useful because a large standard deviation can be investigated by opening the lowest-return child episode and checking its action/reward sequence.

### 16.3 Why trace evaluation instead of all training steps? / 为什么追踪评估而非全部训练？

A 50,000-step run could create tens of thousands of traces, making the project noisy and adding unnecessary serialization and network overhead. The implementation therefore uses:

- W&B metrics for dense training-time scalar logging
- Weave for a small number of structurally rich evaluation episodes
- `max_trace_steps=25`, matching this environment's maximum episode length

This is an **observability design choice（可观测性设计选择）**: use aggregate metrics for scale and detailed traces for diagnosis.

### 16.4 Commands and credentials / 命令与登录

```bash
python -m pip install wandb weave
wandb login

python weave_ver.py --algorithm iql --episodes 5
python weave_ver.py --algorithm cql --episodes 5
```

The default project is `gioiazheng/ai-and-computer-games`. `wandb login` stores credentials for the SDK; alternatively, set `WANDB_API_KEY` in the shell environment. Never write the actual API key into `weave_ver.py`, notes, notebooks, or Git history.

Unlike the Quickstart's Qwen example, this script does not instantiate an OpenAI-compatible client and does not call W&B Serverless Inference. Therefore, tracing these RL evaluations does not create LLM inference requests or token charges.

## 17. Supplied IQL W&B version / 收到的 IQL W&B 课堂版本

Two additional files were supplied after the practical and are preserved
unchanged under `materials/day-03/mpe2-wandb-2026/source/`:

```text
spread_iql_wandb_ver.py  fixed 200,000-step IQL experiment and W&B logging
dqn_wandb_ver.py         DQN components plus detailed training diagnostics
```

These are local course materials, not replacements committed to the maintained
practical. Both `materials/` and `notes/` remain excluded from Git.

Official reference / 官方参考：
[W&B Models Quickstart](https://docs.wandb.ai/models/quickstart)

### 17.1 Quickstart mapped to this code / Quickstart 与代码的对应关系

The official Quickstart describes a small lifecycle:

```text
install and authenticate
 -> initialize a run
 -> store hyperparameters in config
 -> log changing metrics
 -> inspect the run in wandb.ai
```

The supplied script implements the same lifecycle for MARL:

```python
import wandb

run = wandb.init(
    project="marl-simple-spread",
    name="iql-n2-seed-42",
    config={...},
)

run.log({"timestep": timestep, ...})
run.finish()
```

The W&B objects are:

- **Project（项目）**: `marl-simple-spread`, a container for comparable runs.
- **Run（一次运行）**: one IQL training execution, named by algorithm, agent
  count, and seed.
- **Config（配置）**: fixed experimental inputs, including learning rate,
  discount, replay size, epsilon schedule, device, and evaluation settings.
- **History（指标历史）**: values logged repeatedly over `timestep` or
  `episode`.

Authentication follows the official Quickstart:

```bash
python -m pip install wandb
wandb login
```

The API key must be stored through the CLI/keyring or an environment variable,
never inside a Python source file.

### 17.2 Experiment configuration / 实验配置

The supplied run fixes:

| Parameter | Value | Meaning / 含义 |
|---|---:|---|
| seed | 42 | Python, NumPy, PyTorch, replay-buffer, and reset reproducibility |
| total timesteps | 200,000 | number of environment interactions |
| learning rate | \(10^{-3}\) | Adam optimizer step size |
| discount \(\gamma\) | 0.95 | future-reward weight |
| replay capacity | 50,000 | maximum transitions per independent learner |
| batch size | 64 | transitions per DQN update |
| learning starts | 1,000 | random/data-collection warmup |
| target update | every 500 steps | hard copy from online to target network |
| epsilon | 1.0 to 0.05 | exploration range |
| epsilon decay | 50,000 steps | linear exploration schedule |
| W&B log interval | 100 steps | aggregation and upload cadence |
| evaluation interval | 10,000 steps | periodic greedy evaluation |
| evaluation episodes | 20 | fixed-seed games per evaluation |

The linear exploration schedule is

\[
\epsilon_t
=\max\left(
\epsilon_{end},
\epsilon_{start}
-\frac{t}{T_{decay}}
(\epsilon_{start}-\epsilon_{end})
\right).
\]

Before \(t=50{,}000\), epsilon falls linearly from 1.0 to 0.05; afterwards it
stays at 0.05.

### 17.3 Two independent DQN learners / 两个独立 DQN

Each agent receives its own observation \(o_i\), replay buffer, Q-network,
target network, and optimizer:

\[
Q_0(o_0,a_0;\theta_0),
\qquad
Q_1(o_1,a_1;\theta_1).
\]

For learner \(i\), the DQN target is

\[
y_i
=r_i+\gamma(1-d_i)
\max_{a_i'}Q_i^{-}(o_i',a_i';\theta_i^-),
\]

and the temporal-difference error is

\[
\delta_i=y_i-Q_i(o_i,a_i;\theta_i).
\]

The code minimizes the Huber loss

\[
\mathcal L_i
=\frac{1}{B}\sum_{b=1}^{B}
\operatorname{Huber}(\delta_{i,b}).
\]

This remains IQL: W&B observes each learner's optimization but does not combine
the two Q-networks or change action selection.

### 17.4 Metric namespaces and x-axes / 指标分组与横轴

The script explicitly defines metric families:

```python
run.define_metric("train/*", step_metric="timestep")
run.define_metric("exploration/*", step_metric="timestep")
run.define_metric("replay/*", step_metric="timestep")
run.define_metric("eval/*", step_metric="timestep")
run.define_metric("episode/*", step_metric="episode")
```

This matters because W&B otherwise uses its own internal logging step. Here,
training/evaluation charts use actual environment timestep \(t\), while episode
charts use completed game number \(k\).

The slash-separated names also group related panels in the dashboard:

| Namespace | Examples | Question answered / 回答的问题 |
|---|---|---|
| `exploration/*` | epsilon | How much random exploration remains? |
| `replay/*` | each buffer's size | Is enough experience available? |
| `train/*` | loss, Q, TD error, gradient norm | Is DQN optimization numerically healthy? |
| `episode/*` | returns and length | Is behavior improving during exploratory training? |
| `eval/*` | greedy return mean/std | Does the current deterministic policy improve? |

### 17.5 Detailed DQN diagnostics / DQN 诊断指标

`dqn_wandb_ver.py` adds a `return_metrics` option while keeping the original
single-loss return for older scripts. It reports the following per learner.

#### Loss / 损失

`loss` is the batch Huber TD loss \(\mathcal L_i\). The supplied IQL script also
logs the two-agent average

\[
\mathcal L_{mean}=\frac{\mathcal L_0+\mathcal L_1}{2}.
\]

A falling loss only means predictions fit the current bootstrap targets more
closely; it is not proof of a better policy.

#### Mean selected Q / 所选动作 Q 均值

`q_mean` is

\[
\bar Q_i
=\frac{1}{B}\sum_{b=1}^{B}
Q_i(o_{i,b},a_{i,b}).
\]

Because Simple Spread returns negative costs, useful Q-values may remain
negative. The trend and consistency with returns matter more than whether the
number is positive.

#### Maximum absolute Q / 最大绝对 Q

`q_abs_max` is

\[
Q_{abs,max}
=\max_{b,a}|Q_i(o_{i,b},a)|.
\]

A rapid uncontrolled increase can indicate Q-value explosion or unstable
bootstrapping. A large value is a warning signal, not automatically an error;
reward scale and horizon determine the expected magnitude.

#### Mean target Q / TD 目标均值

`target_q_mean` is

\[
\bar y_i=\frac{1}{B}\sum_{b=1}^{B}y_{i,b}.
\]

Comparing \(\bar y_i\) with `q_mean` helps show whether current predictions
systematically lag above or below their targets.

#### Mean absolute TD error / 平均绝对 TD 误差

`td_error_abs_mean` is

\[
\overline{|\delta_i|}
=\frac{1}{B}\sum_{b=1}^{B}|y_{i,b}-Q_i(o_{i,b},a_{i,b})|.
\]

The script also logs the arithmetic mean across the two learners. High TD error
can reflect early learning, a changing teammate, rare transitions, or unstable
targets.

#### Gradient norm / 梯度范数

For network parameters \(\theta=(\theta_1,\ldots,\theta_P)\), the reported
global L2 gradient norm is

\[
\|\nabla_\theta\mathcal L\|_2
=\sqrt{\sum_{p=1}^{P}
\|\nabla_{\theta_p}\mathcal L\|_2^2}.
\]

It is computed after `loss.backward()` and before `optimizer.step()`. The code
only measures the norm; it does not clip or modify gradients. Spikes can reveal
unstable batches even when the averaged loss chart looks smooth.

### 17.6 Why aggregate every 100 steps? / 为什么每 100 步聚合？

The script may perform a DQN update at every environment step after warmup, but
it stores diagnostic values locally and sends their mean every 100 timesteps.
For metric \(m\) in logging window \(W_t\):

\[
\bar m_t
=\frac{1}{|W_t|}\sum_{u\in W_t}m_u.
\]

This reduces network traffic and visual noise. The tradeoff is that a short
single-step spike can be hidden by the mean, which is why maximum-Q and gradient
diagnostics should be interpreted together.

### 17.7 Episode and team reward / Episode 与团队回报

For episode \(k\), each agent return is

\[
R_{i,k}=\sum_{t=0}^{T_k-1}r_{i,t}.
\]

The supplied script defines game/team return as

\[
R_{team,k}=\frac{R_{0,k}+R_{1,k}}{2}.
\]

With `local_ratio=0.0`, MPE2 gives both agents the same shared reward, so
\(R_{0,k}=R_{1,k}\) and the mean equals either agent's return. Using the mean
also remains well-defined if local reward is later introduced; summing would
make the score scale with agent count.

The rolling training score is

\[
\bar R_{100,k}
=\frac{1}{|W_k|}\sum_{j\in W_k}R_{team,j},
\qquad |W_k|\le100.
\]

Training episodes include epsilon-greedy actions, so this is a learning
diagnostic rather than a clean policy score.

### 17.8 Periodic greedy evaluation / 定期贪心评估

Every 10,000 training steps, both networks are evaluated with epsilon 0 on the
same 20 seeds:

\[
s_k=100000+42+k,
\qquad k=0,\ldots,19.
\]

Using fixed evaluation seeds makes checkpoint-to-checkpoint curves less noisy:
each checkpoint sees the same initial conditions. For evaluation returns
\(R_1,\ldots,R_K\), the script logs

\[
\bar R_{eval}=\frac{1}{K}\sum_{k=1}^{K}R_k,
\]

\[
\sigma_{eval}
=\sqrt{\frac{1}{K}
\sum_{k=1}^{K}(R_k-\bar R_{eval})^2}.
\]

Fixed seeds support fair learning-curve comparison, but final reporting should
also include fresh held-out seeds so the policy is not judged only on one small
evaluation set.

### 17.9 Supplied version versus maintained version / 两个版本的区别

| Capability | Supplied `spread_iql_wandb_ver.py` | Maintained `practicals/.../wandb_ver.py` |
|---|---|---|
| Algorithms | IQL only | IQL and centralized Q-learning |
| Configuration | constants in source | command-line arguments |
| Default steps | 200,000 | configurable, default 50,000 |
| W&B modes | online default | online, offline, disabled |
| DQN diagnostics | rich Q/TD/gradient metrics | loss-focused metrics |
| Evaluation | periodic, 20 fixed seeds | final evaluation |
| Metric axes | separate timestep and episode axes | one global-step training axis |
| Evaluation table | no | yes |
| Model artifact | no | yes |
| Checkpoints | final two IQL models | IQL or CQL models plus artifact metadata |
| Import style | expects `dqn2` | imports maintained `dqn.py` directly |

Neither version needs to replace the other. The supplied script is useful for
studying detailed IQL training dynamics; the maintained entry point is better
for repeatable IQL/CQL comparisons and classroom smoke tests.

### 17.10 Run boundary and practical cautions / 运行边界与注意事项

1. **Filename mismatch / 文件名不匹配**: the main script imports
   `from dqn2 import ...`, but the supplied helper is named
   `dqn_wandb_ver.py`. Under the archived filenames, Python raises
   `ModuleNotFoundError: No module named 'dqn2'`. Rename only a working copy or
   change the import in a separate experiment directory.
2. **Online-only default / 默认在线**: there is no `mode="offline"` or disabled
   CLI option. Log in first and expect the run to upload.
3. **Long run / 长训练**: 200,000 steps plus 19 periodic evaluation points can
   take substantial class time. Start with a short copied configuration when
   verifying setup.
4. **Checkpoint timing / 保存时机**: models are saved only after the training
   loop completes normally. An interrupted run can leave W&B history but no
   final `.pth` checkpoints.
5. **No artifact linkage / 没有 Artifact 关联**: checkpoints are local files,
   not W&B model artifacts in this supplied version.
6. **Metric aggregation / 指标聚合**: training diagnostics are 100-step means,
   while episode metrics and periodic evaluation are logged at different
   cadences. Dashboard lines should not be interpreted as one-to-one events.

### 17.11 Minimal conceptual run sequence / 最小运行流程

Preserving the archive unchanged, a separate throwaway working directory would
use:

```bash
conda activate pettingzoo
python -m pip install wandb
wandb login

# In a separate working directory, not in the archived source:
cp dqn_wandb_ver.py dqn2.py
python spread_iql_wandb_ver.py
```

Before committing to 200,000 steps, reduce only the copied script's
`TOTAL_TIMESTEPS`, `EVAL_FREQUENCY`, and `EVAL_EPISODES` for a smoke test. The
archived teacher-supplied files should remain unchanged for provenance.

### 17.12 What to inspect in the dashboard / Dashboard 看什么？

A practical reading order is:

1. `episode/return_100_mean`: is exploratory training improving?
2. `eval/team_return_mean` and `eval/team_return_std`: is greedy policy quality
   improving consistently?
3. `exploration/epsilon`: did changes happen during exploration decay or after
   it reached 0.05?
4. `train/loss_mean` and `train/td_error_abs_mean`: are bootstrap errors
   shrinking or oscillating?
5. per-agent `q_mean`, `q_abs_max`, and `grad_norm`: is one independent learner
   becoming numerically unstable while the other remains healthy?
6. `replay/*_size`: were diagnostics recorded before or after the replay buffer
   filled sufficiently?

No single chart proves that the MARL policy learned cooperation. The strongest
evidence combines stable optimization diagnostics, held-out greedy evaluation,
multiple training seeds, and visual inspection of representative episodes.
