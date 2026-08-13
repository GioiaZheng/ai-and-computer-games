# Day 3 - Imitation Learning / 模仿学习

> Day 3 master guide / Day 3 总讲解：`README.md`

## 中文精读导读 / Detailed bilingual reading guide

本讲的知识演化不是四个缩写的罗列，而是四种不同层次的问题：

- **BC 问“专家在这里做什么？”** 把动作当监督标签，简单、便宜，但容易在陌生状态中连续犯错。
- **IRL 问“什么目标能解释专家这样做？”** 先学习 reward，再用 RL 求 policy，解释性更强但计算昂贵且 reward 不唯一。
- **GAIL 问“怎样让我的整体行为分布像专家？”** 判别器把相似度变成代理 reward，不要求显式写出真正奖励。
- **AIRL 问“能否在对抗式模仿中恢复可迁移的奖励？”** 用结构化 discriminator 分离 reward approximator 与 shaping/value 项。

贯穿全讲的核心区别是 **action matching（动作匹配）** 与 **distribution matching（分布匹配）**。BC 在专家访问过的 state 上逐点匹配 action；GAIL 更关注 learner 长期访问哪些 state-action。IRL 则比较更抽象的 feature expectation，试图让 learner 与 expert 在“重要行为特征”上接近。

初学者应特别注意：没有环境 reward 不等于没有训练目标。BC 使用 cross-entropy；GAIL 使用 discriminator-derived reward；IRL 先从 demonstrations 学 reward。它们改变的是 learning signal 的来源，而不是取消 optimization objective。

Boxing 中可以把差异记成：BC 学“看到这个画面按什么键”；IRL 学“专家是在奖励击中、距离还是防守”；GAIL 学“怎样产生像专家一样的攻防轨迹”；AIRL 希望从这种对抗训练中进一步分离出任务 reward。项目实践应先保证 demonstration 的 observation、action、episode 边界正确，再考虑算法复杂度。

## Lecture roadmap / 本讲主线

Day 3 follows one central question: **how can a learner use expert demonstrations instead of discovering every useful behavior from sparse environment reward?**

```text
Expert demonstrations
    |
    +-> BC: directly clone the expert policy
    |
    +-> IRL: infer a reward, then solve an RL problem
    |
    +-> GAIL: adversarially match expert behavior
             |
             +-> AIRL: adversarial imitation with a recoverable reward structure
```

The shortest professional comparison is:

| Method | Guiding question / 引导问题 | Main learned object / 主要学习对象 | Key difficulty / 关键困难 |
|---|---|---|---|
| BC | What and when should I do? | Policy from expert state-action pairs | Distribution shift |
| IRL | Why does the expert do that? | Reward function, followed by an RL policy | Nested optimization and reward ambiguity |
| GAIL | How can my behavior look expert-like? | Policy matching expert occupancy | Adversarial instability and interaction cost |
| AIRL | Can adversarial learning recover the underlying objective? | Structured reward plus shaping term | Stronger assumptions and greater complexity |

Core transformations:

\[
\text{BC:}\qquad (s,a)_E \rightarrow \pi_\theta(a\mid s),
\]

\[
\text{IRL:}\qquad \pi_E \rightarrow \hat R_E
\rightarrow \text{RL} \rightarrow \pi,
\]

\[
\text{GAIL:}\qquad
\pi \rightarrow \text{agent trajectories}
\rightarrow D \rightarrow \text{proxy reward}
\rightarrow \pi',
\]

\[
\text{AIRL:}\qquad
\text{adversarial imitation}
+ \text{structured, recoverable reward}.
\]

Beginner memory aid / 小白记忆法：

- **BC copies actions / BC 抄动作。**
- **IRL searches for reasons / IRL 猜目标。**
- **GAIL matches behavior / GAIL 学得像。**
- **AIRL tries to recover the reason while matching behavior / AIRL 边学得像，边恢复奖励结构。**

### Formula notation / 公式符号表

| Symbol | Meaning / 含义 |
|---|---|
| \(s_t,a_t,s_{t+1}\) | state, action, and next state at time \(t\) / 当前状态、动作、下一状态 |
| \(T\) | final step or finite trajectory horizon / 轨迹终点或时域长度 |
| \(\gamma\in[0,1)\) | discount factor / 折扣因子 |
| \(\tau=(s_0,a_0,\ldots,s_T)\) | one trajectory / 一条完整轨迹 |
| \(E\) or subscript \(E\) | expert / 专家，例如 \(\pi_E,\tau_E\) |
| \(\pi(a\mid s)\) | probability that policy \(\pi\) selects action \(a\) in state \(s\) / 策略的动作概率 |
| \(R(s,a)\), \(r(s,a)\) | reward function and one reward value / 奖励函数与一次奖励值 |
| \(\phi(s,a)\in\mathbb R^k\) | hand-designed or learned feature vector / 特征向量 |
| \(w\in\mathbb R^k\) | weights of the reward features / 奖励特征权重 |
| \(\mu(\pi)\) | discounted expected feature count under policy \(\pi\) / 策略的折扣期望特征计数 |
| \(\psi^\pi(s)\) | discounted state-visitation frequency / 折扣状态访问频率 |
| \(D(s,a)\in(0,1)\) | discriminator estimate that a sample is expert-like / 判别器认为样本来自专家的概率 |
| \(g_\theta\) | AIRL reward approximator / AIRL 奖励近似网络 |
| \(h_\varphi\) | AIRL shaping or value-like network / AIRL 塑形项网络 |
| \(\mathbb E[\cdot]\) | expectation, an average over possible trajectories / 数学期望 |
| \(\mathbf 1[\cdot]\) | indicator: 1 if the condition is true, otherwise 0 / 指示函数 |

Notation warning / 符号提醒：many papers use \(\phi\) both for an IRL feature function and for AIRL network parameters. To avoid confusion in these notes, **IRL uses \(\phi(s,a)\) for features, while AIRL uses \(\varphi\) for the parameters of \(h_\varphi\)**. They are unrelated objects.

## 1. What is imitation learning? / 什么是模仿学习？

### 课堂内容 / Lecture content

**Imitation Learning (IL，模仿学习)** 也常被称为：

- **Learning from Demonstration (LfD，从示范中学习)**
- **Apprenticeship Learning（学徒式学习）**

系统中有两个主要角色：

- **Expert（专家）**：提供正确或高质量行为示范的人或策略。
- **Learner（学习者）**：观察示范，并学习如何做出相似行为的智能体。

最简单的直觉是：专家做一次，学习者尝试学会“在什么状态下应该采取什么动作”。

Formally, an expert generates demonstrations such as

\[
\tau_E = (s_0, a_0, s_1, a_1, \ldots, s_T),
\]

where \(s_t\) is a **state（状态）**, \(a_t\) is an **action（动作）**, and \(\tau_E\) is an expert **trajectory（轨迹）**. The learner tries to obtain a policy \(\pi\) whose behavior is close to the expert policy \(\pi_E\).

### Boxing example / Boxing 例子

一条专家示范可以包含：

1. 画面中对手向前移动，专家选择后退。
2. 对手出拳后露出空隙，专家选择反击。
3. 回合快结束且比分领先，专家选择保持距离。

示范数据不是一句“赢得比赛”，而是一系列具体的 `(state, action)` 对。

## 2. RL versus imitation learning / 强化学习和模仿学习

### 课堂内容 / Lecture content

**Reinforcement Learning (RL，强化学习)** 从环境提供的 **reward（奖励）** 学习：

\[
s_t \rightarrow a_t \rightarrow r_t, s_{t+1}.
\]

The learner explores the environment and updates its policy according to rewards.

**Imitation Learning** primarily learns from the expert's behavior. It asks whether the learner's behavior matches what the expert did, rather than relying only on a hand-designed environmental reward.

关键区别：

| Question / 问题 | Reinforcement Learning | Imitation Learning |
|---|---|---|
| Main teaching signal / 主要教学信号 | Environment reward / 环境奖励 | Expert demonstrations / 专家示范 |
| What is provided? / 已知什么？ | Reward function or reward samples | State-action examples or trajectories |
| Main difficulty / 主要难点 | Exploration and reward design | Demonstration quality and generalization |
| Simple intuition / 直觉 | “Try, receive feedback, improve.” | “Watch the expert, then behave similarly.” |

两者不是互斥关系。IRL 和 GAIL 都会重新使用 RL 的环境交互和策略优化过程。

## 3. Three major approaches / 三条主要路线

课堂将 IL 分成三类：

1. **Behavioral Cloning (BC，行为克隆)**
2. **Inverse Reinforcement Learning (IRL，逆强化学习)**
3. **Generative Adversarial Imitation Learning (GAIL，生成对抗模仿学习)**

可以用三个问题区分它们：

| Method | Core question / 核心问题 | Learned object / 学什么？ |
|---|---|---|
| BC | What and when should I do? / 什么时候做什么？ | Direct policy \(\pi(a\mid s)\) |
| IRL | Why does the expert do that? / 专家为什么这么做？ | Reward function \(R(s,a)\), then a policy |
| GAIL | How can my behavior look like the expert's? / 怎样让行为分布看起来像专家？ | Policy trained against a discriminator |

## 4. Behavioral Cloning / 行为克隆

### 4.1 Core idea / 核心思想

BC treats imitation as **supervised learning（监督学习）**:

- input \(x\): state \(s\)
- label \(y\): expert action \(a_E\)
- model: policy \(\pi_\theta(a\mid s)\)

For discrete actions, the policy is a classifier whose number of output classes equals the number of possible actions.

例如 Boxing 有 18 个离散动作时，神经网络输出 18 个 **logits（未归一化分类分数）**。经过 softmax 后，每个值可以解释为选择对应动作的概率。

BC searches for a policy with as few expert state-action mismatches as possible:

\[
\hat\pi = \arg\min_{\pi \in \Pi}
\frac{1}{n}\sum_{i=1}^{n}\frac{1}{H}\sum_{h=1}^{H}
\mathbf{1}[\pi(x_h^i) \ne a_h^i].
\]

Term annotations / 术语注释：

- \(\Pi\): **policy class（策略集合）**, all policies the model can represent.
- \(n\): number of expert trajectories / 专家轨迹数量。
- \(H\): **horizon（时域长度）**, number of steps in one trajectory.
- \(\mathbf{1}[\cdot]\): **indicator function（指示函数）**, equal to 1 when the condition is true and 0 otherwise.

The outer \(\arg\min\) means “choose the policy in \(\Pi\) with the smallest average mismatch rate.” The two sums average first over \(H\) time steps and then over \(n\) trajectories.

For neural-network training, the non-differentiable mismatch indicator is normally replaced by cross-entropy. For one expert pair \((s,a_E)\),

\[
\mathcal L_{BC}(s,a_E)
=-\log \pi_\theta(a_E\mid s).
\]

For a batch \(B\),

\[
\mathcal L_{BC}
=-\frac{1}{|B|}\sum_{(s,a_E)\in B}
\log \pi_\theta(a_E\mid s).
\]

Minimizing this loss increases the probability assigned to the expert's action. For example, predicting expert-action probability \(0.8\) gives loss \(-\log(0.8)\approx0.223\), while probability \(0.1\) gives the much worse loss \(-\log(0.1)\approx2.303\).

### 4.2 Training process / 训练步骤

1. Load a batch of expert state-action pairs `(s, a)`.
2. If actions are **one-hot encoded（独热编码）**, convert them to class indices when required by the loss function.
3. Pass each state through the model and obtain action logits.
4. Compare the predicted action distribution with the expert action, commonly using **cross-entropy loss（交叉熵损失）**.
5. **Backpropagate（反向传播）** the loss.
6. Update model **parameters（参数）** with an optimizer such as Adam.

Minimal PyTorch-shaped pseudocode:

```python
states, expert_actions = next(expert_loader)

logits = policy(states)
loss = cross_entropy(logits, expert_actions)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

注意：`logits` 不是概率，不需要先手动做 softmax；PyTorch 的 `cross_entropy` 会在内部完成数值更稳定的处理。

### 4.3 Why BC can fail / BC 为什么会失败？

BC 的主要问题是 **distribution shift（分布偏移）**，也称 **covariate shift（协变量偏移）**。

训练数据来自专家访问过的状态分布 \(d_{\pi_E}(s)\)，但部署时学习者访问的是自己的状态分布 \(d_{\pi}(s)\)。一个很小的动作错误会把学习者带到专家数据中很少出现的状态；模型在该状态上更容易继续犯错，于是误差沿时间累积。

Boxing example:

- 专家通常保持安全距离，所以示范中很少出现“被逼到角落”的画面。
- 学习者一次错误移动后进入角落。
- 训练集没有足够的角落脱困动作，模型可能连续选择错误动作。
- 单步分类准确率很高，也不保证整局策略表现良好。

这解释了课堂表格里的 “Previously unseen state? Panic mode.”

### 4.4 Strengths and limitations / 优缺点

**Strengths:** easy to implement, fully offline training is possible, and large demonstration datasets can work very well.

**Limitations:** no explicit understanding of *why* an action is good; vulnerable to distribution shift; can reproduce expert mistakes and dataset bias.

## 5. Proxy task and proxy reward / 代理任务与代理奖励

### 课堂内容 / Lecture content

回到 RL 以后，需要区分两层目标：

- **Real task（真实任务）**：专家真正想完成的目标。
- **Real reward（真实奖励）**：环境为真实任务提供的奖励。
- **Proxy task（代理任务）**：让学习者模仿专家。
- **Proxy reward（代理奖励）**：由另一个实体给出，用来衡量学习者模仿得有多像。

例如，Boxing 的真实任务可能是赢得比赛，真实奖励可能来自比分变化。GAIL 的代理任务却是“产生与专家相似的状态-动作分布”，代理奖励由 discriminator 给出。

重要风险：优化 proxy reward 不一定等于完成 real task。如果专家示范有限，智能体可能学会表面相似的动作，却没有学到真正取胜的原因。

## 6. Inverse Reinforcement Learning / 逆强化学习

### 6.1 Core idea / 核心思想

IRL does not directly ask the learner to copy every action. Its goal is to infer a reward function \(\hat R_E\) that best explains the expert policy \(\pi_E\), and then use RL to optimize a policy under that learned reward.

流程：

\[
\text{expert demonstrations}
\rightarrow \hat R_E
\rightarrow \text{RL training}
\rightarrow \hat\pi.
\]

相比 BC，IRL 试图回答的是：“专家的行为说明它重视什么？”

### 6.2 Reward representation / 奖励函数表示

课堂中的 skeleton 比喻强调：模型只能学习我们允许它表达的函数。

- **Skeleton（骨架）**：由研究者选择的 reward representation / 奖励函数结构。
- **Meat（填充部分）**：模型从数据中学习到的具体参数。
- Standard RL: humans usually design both the reward structure and its values.
- IRL: humans choose a representational family, then the model estimates its parameters.

如果表示能力过弱，真实奖励不在这个函数族里，模型再努力也无法恢复它。这叫 **model misspecification（模型设定错误）**。

### 6.3 Linear reward and feature count / 线性奖励与特征计数

一种经典 reward skeleton 是加权特征和：

\[
R(s,a) = w_1\phi_1(s,a) + \cdots + w_k\phi_k(s,a)
= w^T\phi(s,a).
\]

- \(\phi_k(s,a)\): the \(k\)-th **feature function（特征函数）**.
- \(\phi(s,a)\): a vector describing properties of the state-action pair that matter.
- \(w_k\): learned importance weight / 学习得到的重要性权重。

Boxing features could include:

\[
\phi(s,a) = [
\text{score difference},
\text{distance to opponent},
\text{distance to ropes},
\text{recent damage},
\text{attack action}
].
\]

If IRL learns a large positive weight for score difference and a large negative weight for being near the ropes, the learned reward says that the expert values scoring while avoiding dangerous positioning.

A **feature count（特征计数）** or discounted feature expectation summarizes how often a policy visits important features:

\[
\mu(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{T}\gamma^t\phi(s_t,a_t)\right].
\]

The result is a \(k\)-dimensional vector, not one scalar: each component accumulates one feature. The factor \(\gamma^t\) gives earlier features more weight; when \(\gamma=1\) over a finite trajectory, it becomes an undiscounted sum.

Classic feature-matching IRL tries to make \(\mu(\pi)\) close to the expert's \(\mu(\pi_E)\). The learner does not need to repeat every exact action if it reaches a similar pattern of important states and actions.

#### State-visitation frequency / 状态访问频率

The photograph introduces \(\psi^{\pi}(s)\), the **state-visitation frequency（状态访问频率）** under policy \(\pi\). Informally, it measures how often the policy reaches state \(s\), with later visits optionally discounted by \(\gamma\).

A standard discounted occupancy definition is:

\[
\psi^{\pi}(s)
= \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty}\gamma^t
\mathbf{1}[s_t=s]\right].
\]

For a known transition model and deterministic policy, an iterative occupancy update can be written using incoming transitions:

\[
\psi_{i+1}^{\pi}(s')
= \psi^0(s')
+ \gamma\sum_{s\in\mathcal S}
\psi_i^{\pi}(s)T(s,\pi(s),s').
\]

- \(\psi^0(s)\): initial-state distribution / 初始状态分布。
- \(T(s,a,s')\): transition probability from \(s\) to \(s'\) after action \(a\).
- \(\gamma\): discount factor / 折扣因子。
- Iterate until the occupancy values converge / 反复迭代直到访问频率收敛。

The slide's notation places the transition and occupancy terms in a different order. The equation above makes the incoming-flow direction explicit: probability mass leaves predecessor states \(s\) and arrives at destination state \(s'\).

The policy's total discounted return can then be expressed through occupancy:

\[
J(\pi)
= \sum_{s\in\mathcal S}
\psi^{\pi}(s)R(s,\pi(s)).
\]

This is a scalar objective for the whole policy. It is related to, but not identical to, the pointwise value function \(V^{\pi}(s)\), which is conditioned on a particular starting state.

#### Expected feature counts / 期望特征计数

Using occupancy, the expected count of feature \(\phi_k\) is:

\[
\mu^{\phi_k}(\pi)
= \sum_{s\in\mathcal S}
\psi^{\pi}(s)\phi_k(s,\pi(s)).
\]

Equivalent trajectory form:

\[
\mu^{\phi_k}(\pi)
= \mathbb{E}_{\pi}\left[
\sum_{t=0}^{\infty}\gamma^t
\phi_k(s_t,a_t)
\right].
\]

The goal is not only to ask **what states** the expert visited, but also **which types of states and how frequently**. Feature-matching IRL therefore tries to make

\[
\mu(\pi) \approx \mu(\pi_E).
\]

Boxing example: two policies might both visit the center and the ropes. If the learner spends 80% of the round trapped near the ropes while the expert only visits briefly, their support overlaps but their state-visitation frequencies and feature counts are very different.

### 6.4 Non-identifiability / 不可辨识性

One expert behavior can often be explained by multiple reward functions. For example, keeping distance in Boxing might mean “avoid damage”, “wait for the opponent to miss”, or “protect a current score lead”. Demonstrations alone may not uniquely identify which reward is the true one.

This is the **non-identifiability（不可辨识性）** problem.

### 6.5 How feature-matching IRL is trained / 特征匹配 IRL 如何训练

The photographed training loop is a nested optimization procedure:

1. Compute the expert feature expectation \(\mu_E=\mu(\pi_E)\) once from the demonstration dataset.
2. Initialize reward weights \(w\).
3. Construct the current reward \(R_w(s,a)=w^T\phi(s,a)\).
4. **Inner loop（内循环）**: train a new RL agent under the current reward.
5. Collect trajectories from that trained policy \(\pi_w\).
6. Estimate learner feature expectations \(\mu(\pi_w)\).
7. **Outer loop（外循环）**: update \(w\) using the difference between expert and learner feature expectations.
8. Repeat until the learner's feature expectations are sufficiently close to the expert's.

A simple conceptual update is:

\[
w \leftarrow w + \alpha\left[
\mu(\pi_E)-\mu(\pi_w)
\right],
\]

where \(\alpha\) is a learning rate. If the expert visits a useful feature more often than the learner, its reward weight is pushed upward; if the learner overuses a feature, its weight is pushed downward.

Component \(j\) is updated as

\[
w_j\leftarrow w_j+\alpha\left[\mu_j(\pi_E)-\mu_j(\pi_w)\right].
\]

Example: if \(\alpha=0.1\), the expert's “stay near center” feature count is \(8\), and the learner's is \(5\), then \(w_j\) increases by \(0.1(8-5)=0.3\).

Why this can be expensive: every outer reward update may require training a fresh or substantially updated RL policy in the inner loop. IRL therefore contains an RL problem inside a reward-learning problem.

## 7. Imitation with RL / 使用 RL 进行模仿

The older lecture material groups several methods under a broader idea: run RL, but construct the training signal from demonstrations rather than relying only on the environment's original reward.

Three useful families are:

1. adversarial imitation / 对抗式模仿
2. support matching / 支持集匹配
3. distribution matching / 分布匹配

### 7.1 GAIL and adversarial imitation / GAIL 与对抗式模仿

GAIL uses a **discriminator（判别器）** to distinguish expert state-action pairs from learner state-action pairs.

- The discriminator learns: “Does this transition look like expert behavior?”
- The policy learns: “Choose actions that make my behavior difficult to distinguish from the expert.”
- A score derived from the discriminator becomes a proxy reward for RL.

The classroom slide maps GAN components to GAIL directly:

| GAN component | GAIL counterpart | 中文解释 |
|---|---|---|
| Real data samples | Expert trajectories | 专家轨迹是真实样本 |
| Generator \(G\) | Policy \(\pi\) | 策略生成智能体行为 |
| Generated samples | Agent trajectories | 当前策略产生的轨迹 |
| Discriminator \(D\) | Expert-vs-agent classifier | 判断轨迹来自专家还是智能体 |
| Generator objective | Fool the discriminator | 让智能体行为看起来像专家 |

Unlike an ordinary GAN generator, the policy does not usually output a complete trajectory in one forward pass. It selects actions sequentially while interacting with the environment; the environment dynamics help generate the trajectory.

Conceptually:

\[
\text{expert data} \longrightarrow D(s,a) \longleftarrow \text{learner rollouts}
\]

\[
D(s,a) \longrightarrow \text{proxy reward} \longrightarrow \text{policy update}.
\]

Unlike plain BC, GAIL repeatedly interacts with the environment and learns from states reached by the current policy. This can reduce the strict train-test mismatch, but training can be unstable because the policy and discriminator change at the same time.

The current classroom slide defines \(D(s,a)\) as the probability that a sample came from the **expert** trajectory. It presents two common reward choices:

\[
r_1(s,a)=\log D(s,a),
\]

or

\[
r_2(s,a)=-\log(1-D(s,a)).
\]

Numeric example / 数值例子：if \(D(s,a)=0.8\), then

\[
r_1=\log(0.8)\approx-0.223,
\qquad
r_2=-\log(0.2)\approx1.609.
\]

If \(D=0.2\), the values become approximately \(-1.609\) and \(0.223\). Under the classroom convention “\(D\) is expert probability,” both rewards therefore rank \(D=0.8\) as more expert-like than \(D=0.2\), although their scales and gradients differ.

Both increase as \(D(s,a)\) approaches 1, so both encourage expert-like behavior. Their numerical behavior differs:

- \(\log D\) is non-positive and strongly penalizes samples that look non-expert.
- \(-\log(1-D)\) is non-negative and grows sharply for highly expert-like samples.
- Implementations clip \(D\) away from exactly 0 and 1 to avoid \(\log(0)\).

Other papers and libraries may define \(D\) as the probability of **policy-generated** data. Under that opposite label convention, the formulas or signs change. Always read the discriminator label definition before interpreting the reward.

Training alternates between two improvements:

1. Update the discriminator so it better separates expert and current-policy behavior.
2. Update the policy so its trajectories better fool the discriminator.

This moving competition explains both GAIL's flexibility and its instability: each model changes the learning target seen by the other model.

#### GAIL training loop / GAIL 训练循环

The photographed implementation sequence is:

1. Collect fresh trajectories from the current policy.
2. Sample expert transitions and policy transitions.
3. Concatenate each observation feature with its action: `[expert_feat, expert_act]` and `[policy_feat, policy_act]`.
4. Concatenate expert and policy examples along the **batch dimension（批次维度）**, normally axis 0.
5. Create binary labels: expert = 1, policy = 0.
6. Pass the combined batch through the discriminator.
7. Compute binary-classification loss against the labels.
8. Backpropagate and update discriminator parameters.
9. Convert discriminator output into imitation reward and update the policy with RL.

PyTorch-shaped discriminator step:

```python
expert_input = torch.cat([expert_features, expert_actions], dim=1)
policy_input = torch.cat([policy_features, policy_actions], dim=1)

disc_input = torch.cat([expert_input, policy_input], dim=0)
labels = torch.cat([
    torch.ones(expert_input.size(0), 1),
    torch.zeros(policy_input.size(0), 1),
], dim=0)

logits = discriminator(disc_input)
disc_loss = binary_cross_entropy_with_logits(logits, labels)
```

Two concatenations serve different purposes:

- `dim=1`: combine features and action into one state-action representation.
- `dim=0`: combine expert and policy examples into one training batch.

For discrete actions, the action may be represented as a one-hot vector or learned embedding before concatenation. A raw integer action index should not usually be treated as a meaningful continuous magnitude.

#### Classroom Atari Alien example / 课堂 Atari Alien 示例

The photographed code setup uses:

- environment: Atari Alien
- raw observation space: `Box(0, 255, (210, 160, 3), uint8)`
- action space: `Discrete(18)`
- expert dataset: 2,048 state-action pairs
- stored expert images: `64 x 64` RGB

This creates a preprocessing contract that must be identical for expert and policy data:

```text
raw 210 x 160 RGB frame
  -> resize or crop to 64 x 64
  -> convert uint8 [0, 255] to float
  -> normalize consistently
  -> arrange channels as required by the encoder
  -> encoder features
  -> concatenate action representation
  -> discriminator
```

If expert images are 64 x 64 but fresh policy frames remain 210 x 160, the discriminator can identify the source from preprocessing artifacts instead of behavior. This is a **shortcut feature（捷径特征）** and makes the learned reward useless.

The 2,048 pairs are also a small demonstration set for a visual control task. A validation split, balanced sampling, data shuffling, and overfitting checks are important.

### 7.2 AIRL / 对抗式逆强化学习

**AIRL (Adversarial Inverse Reinforcement Learning，对抗式逆强化学习)** uses adversarial classification but designs the discriminator so that an underlying reward component can be recovered.

This answers the classroom question: if both GAIL and AIRL use a discriminator, what is the difference?

In ordinary GAIL, the discriminator-derived score is primarily an imitation signal. It can produce expert-like occupancy without yielding a portable, interpretable task reward.

AIRL uses a structured discriminator:

\[
D_{\theta,\varphi}(s,a,s')=
\frac{\exp(f_{\theta,\varphi}(s,a,s'))}
{\exp(f_{\theta,\varphi}(s,a,s'))+\pi(a\mid s)},
\]

with

\[
f_{\theta,\varphi}(s,a,s')=
g_\theta(s,a)+\gamma h_\varphi(s')-h_\varphi(s).
\]

The two learned components have different roles:

- \(g_\theta(s,a)\): **reward approximator（奖励近似器）**, intended to recover the task reward.
- \(h_\varphi(s)\): **shaping/value term（奖励塑形或价值项）**, intended to absorb potential-based shaping and represent a value-like function.
- \(\gamma h_\varphi(s')-h_\varphi(s)\): **potential-based reward shaping（基于势函数的奖励塑形）**.

Read it in two layers / 分两层读：

1. The inner score \(f_{\theta,\varphi}\) adds a task-reward estimate \(g_\theta\) to the change in potential \(h_\varphi\).
2. The outer fraction compares \(\exp(f)\) with the current policy probability \(\pi(a\mid s)\), producing \(D\in(0,1)\).

Here \(\exp(x)=e^x\) makes both terms positive. A larger \(f\) increases \(D\), while a larger current-policy probability in the denominator decreases the discriminator's confidence that the transition is uniquely expert-like.

At the ideal solution, the combined \(f(s,a,s')\) has an advantage-like interpretation:

\[
f(s,a,s') \approx A^*(s,a).
\]

Why separate \(g\) and \(h\)? Two rewards can induce similar optimal behavior while differing by a shaping term. AIRL tries to put transferable task preference into \(g\) and state-dependent shaping into \(h\), instead of mixing everything into one opaque discriminator score.

| Aspect | GAIL | AIRL |
|---|---|---|
| Primary goal | Match expert occupancy/behavior | Recover a reward while imitating |
| Discriminator | General expert-vs-policy classifier | Structured with reward and shaping components |
| Learned signal | Useful proxy reward | Reward component intended to transfer |
| Inputs shown here | Usually \((s,a)\) | \((s,a,s')\) and current policy probability |
| Main limitation | Reward may be policy/dynamics-specific | More assumptions and implementation complexity |

Recoverability depends on modeling assumptions and sufficient data; AIRL does not guarantee that a finite demonstration dataset reveals the unique “true human reward”.

### 7.3 Support matching / 支持集匹配

The **support（支持集）** of expert data is the region of state-action space covered by expert demonstrations. A simple support-matching reward is high when the learner visits a state-action pair near that region:

\[
r(s,a) \approx 1 - \min_{(s_E,a_E)\in D_E}
d\big((s,a),(s_E,a_E)\big)^2.
\]

Here, \(d\) is a distance function and \(D_E\) is the expert dataset.

直觉：如果学习者当前行为很接近某个专家行为，就获得较高代理奖励。这种方法容易接入现有 RL 算法，但有两个限制：

- “接近专家访问过的区域”不等于正确复现专家访问各区域的频率。
- 距离函数本身决定了什么叫“相似”，不合理的表示会产生误导奖励。

### 7.4 Distribution matching / 分布匹配

**Distribution matching（分布匹配）** asks the learner to match the expert's marginal or occupancy distribution over state-action pairs:

\[
\min_{\pi} d\big(\rho_{\pi}(s,a),\rho_E(s,a)\big),
\]

where \(\rho_{\pi}\) is the learner's **occupancy measure（占用度量）**, describing how frequently policy \(\pi\) visits each state-action pair.

One possible distance is the **Wasserstein distance（Wasserstein 距离）**, also called the **Earth Mover's Distance（推土机距离）**. It imagines moving probability mass from the learner distribution to the expert distribution and measures the minimum transport cost.

**PWIL (Primal Wasserstein Imitation Learning)** turns this matching cost into a reward for RL. Compared with simple support matching, it cares not only whether a learner transition lies near expert data, but also how the overall probability mass is allocated.

Tradeoffs:

- It captures trajectory-level or distribution-level similarity more faithfully.
- Full-trajectory matching can be slower and more memory intensive.
- Greedy matching can make locally convenient choices that hurt the global match.
- Matching the entire expert distribution may be unnecessary when several different behaviors solve the task equally well.

## 8. Comparison / 方法对比

| Method | Direct target | Reward | Environment interaction | Main drawback |
|---|---|---|---|---|
| BC | Expert action | Not required for BC training | Not required during offline training | Distribution shift |
| IRL | Expert reward | Learned explicitly | Required for subsequent RL | Slow, complex, non-identifiable reward |
| GAIL | Expert-like behavior distribution | Proxy reward from discriminator | Repeatedly required | Unstable and environment-sample intensive |
| AIRL | Recoverable reward plus expert-like behavior | Structured adversarial reward | Repeatedly required | More assumptions and implementation complexity |
| Support matching | Expert data support | Distance-based proxy reward | Required | Ignores expert visitation frequencies |
| Distribution matching | Expert occupancy distribution | Distribution-distance reward | Required | Slower and sensitive to matching design |

## 9. Evolution of the ideas / 方法演化脉络

### 理解补充 / Learning supplement

The evolution is driven by what plain supervised imitation fails to capture:

1. **Behavioral Cloning** directly copies actions. It is simple, but errors change the states the learner sees.
2. **Interactive data aggregation methods such as DAgger** address distribution shift by letting the learner visit states and asking the expert for the correct action there.
3. **Inverse Reinforcement Learning** moves from copying actions to recovering the objective behind those actions.
4. **GAIL and related adversarial methods** skip an explicit standalone reward-recovery stage and train the policy using a learned discriminator signal.
5. **AIRL** reintroduces reward recovery through a deliberately structured adversarial discriminator.
6. **Distribution-matching methods** generalize the idea further: match the learner's occupancy or trajectory distribution to the expert's distribution using a chosen distance.

This is not a simple “new method always replaces old method” story. BC remains a strong baseline when demonstrations are abundant; IRL is useful when the reward itself matters or must transfer; GAIL is useful when environment interaction is available and behavior-level matching is more important than an interpretable reward.

## 10. Boxing project connection / 与 Boxing 项目的关系

For the current project, the safest progression is:

1. Train and evaluate the DQN baseline from environment rewards.
2. Define a demonstration format containing observations, actions, episode boundaries, and optional rewards.
3. Train a BC baseline from demonstrations.
4. Compare BC and DQN using the same evaluation protocol.
5. Only consider GAIL or IRL after the demonstrations and evaluation pipeline are reliable.

Useful evaluation metrics:

- match win rate / 比赛胜率
- mean episode return / 平均回合回报
- score difference / 比分差
- action agreement with expert on held-out states / 在留出状态上的专家动作一致率
- performance after the policy reaches unfamiliar states / 陌生状态下的恢复能力

## 11. Quick glossary / 术语速查

| Term | 中文 | Plain-language meaning / 简明解释 |
|---|---|---|
| Demonstration | 示范数据 | A trajectory produced by an expert |
| Expert policy | 专家策略 | The rule used by the expert to select actions |
| Learner policy | 学习者策略 | The policy being trained |
| Behavioral Cloning | 行为克隆 | Supervised prediction of expert actions |
| Inverse RL | 逆强化学习 | Infer a reward that explains expert behavior |
| GAIL | 生成对抗模仿学习 | Train a policy to fool an expert-behavior discriminator |
| Logit | 未归一化分类分数 | Raw model output before softmax |
| Distribution shift | 分布偏移 | Deployment states differ from training states |
| Proxy reward | 代理奖励 | A substitute signal used to optimize imitation |
| Feature function | 特征函数 | Maps a state or state-action pair to a meaningful number |
| Feature count | 特征计数 | Accumulated occurrence of important features in a trajectory |
| Non-identifiability | 不可辨识性 | Several rewards can explain the same behavior |

## 12. Self-check questions / 自测题

1. Why is BC a supervised-learning problem even though actions are sequential?
2. Why can high action accuracy on expert data still produce a weak game-playing policy?
3. What is the difference between a real reward and a proxy reward?
4. In IRL, what is selected by the researcher and what is learned from demonstrations?
5. What does \(w^T\phi(s,a)\) mean in a linear reward model?
6. Why does GAIL require environment interaction while offline BC does not?
7. Which approach would you choose for Boxing if only a small fixed demonstration dataset were available, and what failure mode would you monitor?

## Source boundary / 来源边界

The main structure and current terminology were reconstructed from the live classroom photographs and the 18-page 2026 UEF lecture supplied on Day 3. The 2024 imitation-learning PDF was used as a supplementary source for details such as support matching, distribution matching, and PWIL. When the sources differ, the current 2026 material takes priority. The DAgger connection, expanded explanations, and Boxing examples are learning supplements added to make the note self-contained.

The companion note `imitation-learning-code-study.md` analyzes the supplied
`bc_agent.py`, `irl_agent.py`, `gail_agent.py`, `utils.py`, and
`bc_model_example.pt`. It keeps the original teaching code separate from
formula corrections, runtime caveats, and recommendations for reproducible
experiments.
