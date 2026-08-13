# Day 1 Complete Guide / 第一天完整讲解

## 0. Roadmap and source boundary / 学习路线与资料边界

Day 1 builds the mathematical and engineering foundation needed by every later
topic: machine learning objectives, gradient-based optimization, neural
networks, the agent-environment loop, Markov decision processes, value
functions, Q-learning, DQN, policy gradients, and PPO. The afternoon research
talks then show where these tools fail under domain shift, adversarial pressure,
missing action labels, and expensive real interaction.

第一天的核心不是记住算法名称，而是建立一条完整因果链：**数据或交互产生训练信号，loss/objective 衡量错误，gradient 更新参数，policy 改变行为，evaluation 检查是否真正完成任务。** 后续 MARL、模仿学习和 Boxing 都是在这条链上增加新的数据来源、目标或结构。

Sources / 资料来源：

- `2026_intro_lecture.pdf`: current course framing and Boxing project.
- `rl_algorithms.pdf`: DQN, policy gradient, actor-critic, and PPO taxonomy.
- `introduction_to_reinforcement_learning_2025.pdf`: MDP, return, value, and Q.
- `automatic_differentiation_survey.pdf`: forward/reverse automatic differentiation.
- `gradient_descent_regression.ipynb`: runnable regression and optimization examples.
- `research-presentations.md`: four course-team research presentations.

## 1. Machine-learning paradigms / 机器学习范式

### 1.1 Supervised learning / 监督学习

Supervised learning receives labelled pairs \((x_i,y_i)\) and learns a mapping
\(f_\theta(x)\) that predicts \(y\). Regression predicts a continuous value;
classification predicts a discrete class.

监督学习的数据同时包含输入与正确答案。线性回归预测连续数值，逻辑回归或神经分类器预测类别。Day 3 的 Behavioral Cloning 本质也是监督学习：state/observation 是 \(x\)，专家 action 是 \(y\)。

\[
\theta^*=\arg\min_\theta
\frac1N\sum_{i=1}^N\ell(f_\theta(x_i),y_i).
\]

### 1.2 Unsupervised and self-supervised learning / 无监督与自监督学习

Unsupervised learning searches for structure without explicit human labels,
such as clusters or compact representations. Self-supervised learning creates
targets from the data itself, for example predicting a masked token or a future
frame.

无监督学习没有人工给定的“正确动作”；自监督学习则从数据内部构造监督信号。它们常用于预训练 encoder，再把表示交给 RL 或 IL。区别在于 representation objective 不等于最终游戏目标，因此仍需下游评估。

### 1.3 Reinforcement learning / 强化学习

RL learns from sequential interaction. The action changes the next observation
and therefore changes future training data. This makes RL different from a
fixed supervised dataset.

强化学习中，模型的输出会反过来改变它接下来看到的数据。一次动作可能立即没有收益，却影响很久之后的结果，因此必须处理 exploration、delayed reward 和 credit assignment。

## 2. Regression, prediction, and loss / 回归、预测与损失

A one-dimensional linear model is

\[
\hat y=f_{a,b}(x)=ax+b.
\]

For samples \((x_i,y_i)\), mean squared error is

\[
\mathcal L(a,b)=\frac1N\sum_{i=1}^N(ax_i+b-y_i)^2.
\]

\(a\) controls slope, \(b\) controls intercept, and the residual
\(e_i=\hat y_i-y_i\) measures one prediction error. Squaring makes positive and
negative errors contribute positively and penalizes large errors strongly.

中文理解：模型是一条由参数控制的直线，loss 把所有样本的预测误差压缩成一个可优化的标量。训练不是“让 loss 看起来小”这么简单，而是在给定模型族中寻找最适合数据的参数；有噪声时，学到的参数通常不会精确等于生成数据时的真实参数。

Some texts use \(1/(2N)\) instead of \(1/N\). The factor \(1/2\) cancels the 2
from differentiating a square and does not change the minimizer. / 有些公式多一个 \(1/2\)，只是让导数更整洁，不改变最优解位置。

### Worked example / 数值例子

Let \(a=2,b=1\) and one sample be \((x,y)=(3,8)\). Prediction is 7, residual is
-1, and squared error is 1. If prediction were 4, squared error would be 16,
showing why MSE reacts strongly to large mistakes.

若训练数据含极端 outlier，MSE 可能被少数样本主导；这也是 DQN 常使用 Huber loss 而非纯 MSE 的原因之一。

## 3. Gradient descent / 梯度下降

The gradient contains one partial derivative per parameter and points toward
the steepest local increase of the loss. Gradient descent moves in the opposite
direction:

\[
\theta_{t+1}=\theta_t-\eta\nabla_\theta\mathcal L(\theta_t).
\]

\(\eta\) is the learning rate. Too small is slow; too large can oscillate or
diverge. The gradient says a local direction, not the distance to the global
optimum.

中文拆解：先在当前参数位置计算 loss 对每个参数的敏感程度，再沿减少 loss 的方向走一小步。学习率决定一步多远。训练曲线 zigzag 可能表示学习率相对曲率过大；loss 完全不动可能是学习率太小、梯度为零、数据管线错误或参数根本没进入 optimizer。

For linear regression:

\[
\frac{\partial\mathcal L}{\partial a}
=\frac2N\sum_i(ax_i+b-y_i)x_i,
\qquad
\frac{\partial\mathcal L}{\partial b}
=\frac2N\sum_i(ax_i+b-y_i).
\]

The notebook visualizes the fitted line, loss curve, and path on a loss
contour. / notebook 的三幅图分别回答“模型现在长什么样”“loss 是否下降”“参数在 loss surface 上如何移动”。

## 4. Logistic regression and cross-entropy / 逻辑回归与交叉熵

Binary logistic regression maps a linear score to a probability:

\[
z=w^\top x+b,
\qquad
p(y=1\mid x)=\sigma(z)=\frac1{1+e^{-z}}.
\]

Binary cross-entropy is

\[
\mathcal L_{BCE}
=-\frac1N\sum_i
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
\]

若真实标签为 1，模型给 \(p=0.9\) 时 loss 很小；给 \(p=0.01\) 时 loss 很大。Cross-entropy 不只关心分类对错，还关心错误时有多自信。多分类 BC 使用 softmax cross-entropy，GAIL discriminator 使用二分类形式。

**Logit（未归一化分数）** is \(z\), not a probability. PyTorch losses such as
`CrossEntropyLoss` and `BCEWithLogitsLoss` expect logits and perform stable
normalization internally. / 不应先手动 softmax/sigmoid 后再喂给这些 `WithLogits` 或 cross-entropy API。

## 5. Neural networks as function approximators / 神经网络作为函数近似器

A layer computes

\[
h=\phi(Wx+b),
\]

where \(W,b\) are learned parameters and \(\phi\) is a nonlinear activation.
Stacking nonlinear layers lets the model represent functions that one linear
map cannot.

神经网络不是“自动懂游戏”的黑盒，它只是可微函数族。CNN 利用局部连接和参数共享处理图像；MLP 适合固定长度向量；recurrent/transformer structures 处理历史。架构先验决定模型容易表达什么，也决定计算和数据需求。

For a discrete-action Q-network:

\[
f_\theta(s)=
[Q_\theta(s,a_0),\ldots,Q_\theta(s,a_{A-1})].
\]

输出维度等于动作数。网络不直接输出“按左键”，而是输出各动作长期价值，再由 argmax 或 exploration rule 选择动作。

## 6. Automatic differentiation and backpropagation / 自动微分与反向传播

Automatic differentiation (AD) applies the chain rule to elementary operations
recorded in a computation graph. It is different from symbolic differentiation
and finite differences.

自动微分不是用很小的 \(\epsilon\) 猜导数，也不是把公式化简成符号表达式。程序在 forward 中记录运算依赖，backward 按 chain rule 高效计算参数梯度。

For \(y=f(g(x))\):

\[
\frac{dy}{dx}=\frac{df}{dg}\frac{dg}{dx}.
\]

- Forward mode propagates derivatives with values and is attractive when inputs
  are few and outputs many.
- Reverse mode propagates adjoints backward and is attractive when a scalar loss
  depends on many parameters; this is neural-network backpropagation.

反向模式正适合“几百万参数 → 一个 scalar loss”。`loss.backward()` 计算梯度，`optimizer.step()` 才真正修改参数，`optimizer.zero_grad()` 防止默认梯度累加。

## 7. Agent-environment loop / 智能体与环境循环

```text
observation -> policy -> action -> environment
      ^                         |
      `---- reward, next obs ---'
```

At time \(t\), the agent observes \(o_t\), selects \(a_t\), receives \(r_t\),
and observes \(o_{t+1}\). A trajectory is

\[
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots).
\]

中文关键点：observation 是 agent 实际看到的信息，state 是理论上的完整环境信息；两者在完全可观测 MDP 中可以相同，在 Atari 像素或 MARL 中常不同。Policy 是决策规则，environment transition 不由 agent 直接控制。

## 8. MDP / 马尔可夫决策过程

An MDP is commonly written

\[
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma).
\]

- \(\mathcal S\): states / 状态集合
- \(\mathcal A\): actions / 动作集合
- \(P(s'\mid s,a)\): transition probability / 转移概率
- \(R(s,a,s')\): reward / 奖励
- \(\gamma\in[0,1)\): discount factor / 折扣因子

The Markov property says the current state contains all information needed for
the future transition:

\[
P(s_{t+1}\mid s_0,a_0,\ldots,s_t,a_t)
=P(s_{t+1}\mid s_t,a_t).
\]

马尔可夫性不是说“未来与过去无关”，而是说过去对未来的影响已经被当前 state 总结。如果单帧 Boxing 看不出拳头运动方向，单帧 observation 就不满足这一信息需求，因此实践中堆叠多帧近似补足速度信息。

When \(P\) and \(R\) are known and the state space is manageable, dynamic
programming can solve Bellman equations. Model-free RL is needed when these
quantities are unknown or too large to enumerate. / 已知转移概率不等于一定要用采样 RL；可计算时可直接做 value iteration 或 policy iteration。

## 9. Reward, return, and discounting / 奖励、回报与折扣

Immediate reward is one-step feedback. Return is future cumulative reward:

\[
G_t=\sum_{k=0}^{T-t}\gamma^k r_{t+k}.
\]

折扣因子有三个作用：降低遥远未来的不确定影响、让无限和可能收敛、表达时间偏好。\(\gamma=0\) 只看当前奖励；接近 1 更重视长期结果。它不是“未来奖励一定不重要”的固定真理，而是任务建模选择。

**Reward design / 奖励设计：** intended goal、proxy metric、training reward 和 evaluation metric 必须区分。Goodhart's Law 提醒我们，优化器可能找到提高 proxy 但违背真实意图的行为，即 reward hacking/specification gaming。

Boxing example: 只奖励“出拳次数”会鼓励乱挥拳；按击中得分更接近目标，但仍可能学会拖延或利用特定对手。最终应评估胜率、分差、多种对手和 seeds，而不是只看训练 reward。

## 10. Value, Q-value, and Bellman equations / V、Q 与 Bellman 方程

State value under policy \(\pi\):

\[
V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s].
\]

Action value:

\[
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
\]

\(V\) 回答“这个状态整体有多好”，\(Q\) 回答“在这里先做某个动作有多好”。离散动作控制通常更方便用 Q，因为可直接比较动作。

Bellman expectation equation:

\[
Q^\pi(s,a)=
\mathbb E\left[r+\gamma
\mathbb E_{a'\sim\pi}[Q^\pi(s',a')]ight].
\]

Bellman optimality equation replaces the next-policy expectation with a max:

\[
Q^*(s,a)=\mathbb E[r+\gamma\max_{a'}Q^*(s',a')].
\]

Bellman equation 是递归一致性条件：当前长期价值必须等于当前奖励加下一状态长期价值。Q-learning 与 DQN 的 TD target 就来自这里。

## 11. Q-learning and DQN / Q-learning 与深度 Q 网络

Tabular Q-learning update:

\[
Q(s,a)\leftarrow Q(s,a)+\alpha
\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right].
\]

括号中是 TD error。表格适合小型离散 state；图像状态无法枚举，因此 DQN 用 \(Q_\theta(s,a)\) 替代表格，并最小化预测与 target 的差。

Stable DQN uses:

- replay buffer / 经验回放，打散时间相关性；
- target network / 目标网络，稳定 bootstrap target；
- epsilon-greedy / 探索与利用；
- preprocessing and frame stacking / 降维并补充运动信息；
- Huber loss and gradient clipping / 降低异常 TD error 影响。

\[
y^{DQN}=r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a').
\]

`(1-d)` prevents bootstrapping after a true terminal transition. / episode 自然结束后不存在未来回报，不能继续引用 next-state Q。

## 12. On-policy, off-policy, and offline RL / 同策略、异策略与离线 RL

- **On-policy:** learns about the same policy that generated the latest data.
  PPO is a common example.
- **Off-policy:** can learn a target policy from data generated by another
  behavior policy. Q-learning/DQN are off-policy.
- **Offline RL:** learns only from a fixed dataset without further environment
  collection.

中文区别：off-policy 不等于 offline。DQN 使用 replay 中旧 policy 的数据，所以是 off-policy，但训练期间仍不断与环境交互，因此不是纯 offline RL。BC 是固定示范上的 offline supervised learning，却不属于标准 reward-based offline RL。

## 13. Model-free and model-based RL / 无模型与基于模型的 RL

Model-free methods learn value or policy without explicitly learning
\(P(s'\mid s,a)\). Model-based methods use a known or learned dynamics model for
planning or imagined rollouts.

无模型不表示“没有神经网络”，而是没有显式环境 dynamics model。Dreamer 类方法学习 world model，在 latent space 想象未来；好处是提高数据效率，风险是 policy 利用模型误差。

## 14. Policy gradient and actor-critic / 策略梯度与 Actor-Critic

Policy-gradient methods optimize a stochastic policy directly:

\[
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G_0],
\qquad
\nabla_\theta J
=\mathbb E[\nabla_\theta\log\pi_\theta(a\mid s)\,A(s,a)].
\]

The actor is the policy; the critic estimates value and supplies a lower-
variance learning signal. Advantage

\[
A(s,a)=Q(s,a)-V(s)
\]

measures whether an action is better than the state's baseline expectation.

中文直觉：如果某动作结果比当前状态平均水平好，就提高它的 log-probability；若更差则降低。Critic 不是对手，而是给 actor 提供评价的价值估计器。

## 15. PPO / 近端策略优化

PPO limits how far the policy moves in one update. With probability ratio

\[
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{old}}(a_t\mid s_t)},
\]

the clipped objective is

\[
L^{CLIP}=\mathbb E_t\left[
\min(r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)
\right].
\]

PPO 的 clip 不是把 reward 截断，而是限制新旧 policy 对已采样动作的概率比，防止一次更新过猛。它仍需要 advantage estimation、value loss、entropy bonus 和可靠 rollout。

## 16. Exploration and evaluation / 探索与评估

Exploration gathers information; exploitation uses current knowledge. An
untried action cannot be known to be bad before evidence is collected.

训练与评估必须分开：训练可使用 epsilon 或 stochastic policy；evaluation 应固定协议，通常关闭随机探索或明确保留多少随机性。至少报告多个 seeds、mean、standard deviation、score difference、win rate 和失败案例。

## 17. Evolution and connections / 演化与课程连接

```text
linear model -> neural function approximator
fixed labelled data -> sequential interaction
known Bellman table -> DQN approximation
value-only method -> actor-critic and PPO
single policy -> MARL, imitation, and adversarial settings
```

第一天学到的 loss、gradient、network 和 evaluation 在后面不会消失：IQL 只是多个 DQN；BC 只是序列数据上的分类；GAIL 组合 discriminator 与 PPO；W&B 记录的仍是 loss、return、gradient 和配置。

Research talks / 研究报告：`research-presentations.md` 详细介绍 domain generalization、catastrophic forgetting、adversarial RL、VPT/inverse dynamics 和 imagined GAIL。这些报告共同说明：平均训练分数不够，还要检查 domain shift、adaptive opponent、label ambiguity 和 model error。

## 18. Common misconceptions / 常见误区

1. Loss lower does not automatically mean task performance higher. / loss 更低不自动代表游戏更强。
2. Reward is immediate; return is cumulative. / reward 与 return 不是同一个量。
3. A negative reward can still be an improvement if it becomes less negative. / 负奖励变得接近 0 也可能是进步。
4. Off-policy is not the same as offline. / 异策略不等于离线。
5. Backprop computes gradients; the optimizer applies updates. / backward 不直接修改参数。
6. Q-values are expected returns, not action probabilities. / Q 值不是动作概率。
7. A model-free agent still has a policy/value model; it lacks an environment model. / 无模型指无 dynamics model。

## 19. Professional glossary / 专业术语

| Term | 中文 | Meaning |
|---|---|---|
| Objective | 目标函数 | Quantity optimized by training |
| Loss | 损失 | Differentiable error minimized in learning |
| Gradient | 梯度 | Local parameter sensitivity of the loss |
| Learning rate | 学习率 | Update step scale |
| Generalization | 泛化 | Performance on unseen data or situations |
| Policy | 策略 | Rule or distribution for selecting actions |
| Return | 累计回报 | Discounted future reward sum |
| Value function | 价值函数 | Expected return from a state |
| Q-function | 动作价值函数 | Expected return after a state-action choice |
| Bootstrap | 自举 | Build a target from another current estimate |
| TD error | 时序差分误差 | Bellman target minus current prediction |
| Exploration | 探索 | Gather information through non-greedy actions |
| Advantage | 优势函数 | Action value relative to state baseline |
| World model | 世界模型 | Learned predictor of environment dynamics |

## 20. Self-check with short answers / 自测与简答

1. **Why subtract the gradient? / 为什么减梯度？** It points uphill; subtraction moves locally downhill. / 梯度指向局部上升最快方向。
2. **Why stack Atari frames? / 为什么堆帧？** One frame lacks velocity/direction information. / 单帧缺少运动方向。
3. **Why is DQN off-policy? / 为什么 DQN 是异策略？** It learns a greedy target from replay generated by older exploratory policies.
4. **What does \(\gamma\) change? / gamma 改变什么？** The relative weight of delayed rewards and effective planning horizon.
5. **Why separate training and evaluation? / 为什么分开？** Exploration noise and moving targets can hide actual greedy-policy quality.
