# Day 2 Complete Guide / 第二天完整讲解

## 0. Roadmap / 学习路线

Day 2 moves from one learning agent to several simultaneously acting agents.
The morning introduces MARL and strategic games; the Minecraft lecture expands
the meaning of an agent from a short reactive controller to a long-horizon,
language-conditioned, memory-using system.

第二天的主线是：当其他 agent 也会观察、行动和学习时，单智能体 MDP 的哪些假设失效？随后 Minecraft 把问题扩大到长时域、层级技能、语言、记忆和开放式评价。

Sources / 资料来源：`multi_agent_reinforcement_learning.pdf`, the
Diplomacy/Backstabbr classroom example, Anssi's Minecraft lecture and reading
list, and the course MARL book.

## 1. From MDP to stochastic game / 从 MDP 到随机博弈

A stochastic game generalizes an MDP to \(N\) agents:

\[
\mathcal G=
(\mathcal S,\{\mathcal A_i\}_{i=1}^N,
P,\{R_i\}_{i=1}^N,\gamma).
\]

At state \(s\), agents choose a joint action

\[
\mathbf a=(a_1,\ldots,a_N)
\in\mathcal A_1\times\cdots\times\mathcal A_N.
\]

The environment moves according to \(P(s'\mid s,\mathbf a)\), and each agent
receives \(R_i(s,\mathbf a,s')\).

中文关键点：下一状态不再只由“我的动作”决定，而由所有 agent 的动作组合决定。每个 agent 的 reward 也可以不同。因此把其他 agent 简单当作静态环境，会遇到非平稳性。

## 2. Cooperative, competitive, and mixed settings / 合作、竞争与混合任务

- **Fully cooperative / 完全合作:** agents share the same return, as in Simple Spread with `local_ratio=0`.
- **Zero-sum competitive / 零和竞争:** one agent's gain is the other's loss, approximately like Boxing score difference.
- **General-sum / 一般和:** interests partly align and partly conflict, as in negotiation or Diplomacy.

合作不表示动作相同，竞争也不表示没有合作可能。Diplomacy 中玩家会临时结盟再改变策略，是 general-sum、communication 和 opponent modelling 的典型例子。

## 3. Partial observability and POSG / 部分可观测与 POSG

In a Partially Observable Stochastic Game, agent \(i\) receives observation
\(o_i\sim O_i(\cdot\mid s)\) rather than the complete state. Its policy is

\[
\pi_i(a_i\mid h_i),
\]

where \(h_i\) may contain observation-action history.

中文理解：两个 agent 位于同一个 world state，却可能看到完全不同的信息。若当前 observation 不足以判断隐藏状态，memory/recurrent network 可用历史近似 belief。把完整 state 偷偷给执行 policy 会造成 information leakage，使实验不符合真实部署条件。

## 4. Joint policy and value / 联合策略与价值

If policies act independently conditioned on local information, a factored
joint policy is

\[
\boldsymbol\pi(\mathbf a\mid s)
=\prod_{i=1}^N\pi_i(a_i\mid o_i).
\]

Agent \(i\)'s value depends on all policies:

\[
V_i^{\boldsymbol\pi}(s)
=\mathbb E_{\boldsymbol\pi}[G_i\mid s_0=s].
\]

单智能体写 \(V^\pi\) 已足够；MARL 必须记住“我的回报取决于队友/对手策略”。同一个动作面对不同 opponent 可能价值完全相反。

## 5. Best response and Nash equilibrium / 最佳响应与纳什均衡

A best response for agent \(i\) maximizes its return against fixed opponent
policies \(\pi_{-i}\):

\[
BR_i(\pi_{-i})
\in\arg\max_{\pi_i}J_i(\pi_i,\pi_{-i}).
\]

A Nash equilibrium \(\boldsymbol\pi^*\) satisfies

\[
J_i(\pi_i^*,\pi_{-i}^*)
\ge J_i(\pi_i,\pi_{-i}^*)
\quad\text{for every }i,\pi_i.
\]

纳什均衡表示在其他策略固定时，没有任何单个 agent 愿意单方面改变。它不是“全局最好”“公平”或“合作成功”的同义词，也可能存在多个均衡。

For a two-player zero-sum game, minimax reasoning is

\[
\max_{\pi_1}\min_{\pi_2}J_1(\pi_1,\pi_2).
\]

Boxing agent 若只针对 random opponent 训练，学到的是对 random 的 best response，不一定对强对手稳健。Self-play 和 opponent pools 用来扩大训练对手分布。

## 6. Core MARL difficulties / MARL 核心困难

### 6.1 Non-stationarity / 非平稳性

Other agents update their policies, so the effective transition distribution
seen by one learner changes over time. Replay may mix experiences generated
against many historical teammate/opponent policies.

其他 agent 也是会动的学习目标。昨天“右移有效”的统计规律，今天可能因对手改变而失效。这会破坏普通 DQN 假设的数据稳定性。

### 6.2 Credit assignment / 信用分配

With one team reward, which agent/action caused success?

共享 reward 能对齐团队目标，却没有直接告诉每个 agent 自己贡献多少。Counterfactual baseline、value decomposition 或 centralized critic 都在尝试提供更细的责任信号。

### 6.3 Scalability / 扩展性

With \(m\) actions per agent, explicit joint actions grow as

\[
|\mathcal A_{joint}|=m^N.
\]

5 个动作、2 个 agent 是 25 种组合，6 个 agent 已是 15,625 种。Joint observation 往往线性增长，joint action 枚举则指数增长。

### 6.4 Coordination and equilibrium selection / 协调与均衡选择

Even cooperative games may have several good conventions. Independent learners
can fail because each waits for the other to coordinate differently.

例如两个 agent 应分别覆盖两个 landmark，但“谁去左边”有两个对称方案。没有身份、通信或随机破缺时，学习可能在方案间摇摆。

### 6.5 Evaluation against populations / 面向对手群体的评估

One opponent and one seed do not measure robustness. Report performance against
random, scripted, historical checkpoints, self-play peers, and held-out agents.

只对训练对手获胜可能是过拟合。竞技 agent 应记录 win rate、score difference、不同 opponent 的分位数和最差案例。

## 7. Independent learning / 独立学习

Independent Q-Learning gives each agent its own Q-function:

\[
Q_i(o_i,a_i;\theta_i).
\]

Each learner stores local transition
\((o_i,a_i,r_i,o_i',d_i)\) and applies a normal DQN update. It scales better than
enumerating all action combinations but treats changing agents as part of the
environment.

IQL 简单、可分散执行，是重要 baseline；缺点是不能直接评价动作组合，也没有显式解决非平稳性和 credit assignment。Day 3 practical 用 Simple Spread 具体实现这一算法。

## 8. Centralized learning / 集中式学习

A centralized Q-function can use joint information:

\[
Q_{central}(o_1,\ldots,o_N,a_1,\ldots,a_N).
\]

它可以直接比较“agent 0 左移且 agent 1 右移”与“两者都左移”，协调表达更强；但 joint action 爆炸，执行时还依赖全局信息。

This is distinct from **CTDE**. Centralized Training with Decentralized
Execution allows extra global information in training while each deployed
actor uses only \(o_i\).

CTDE 的常见结构是 centralized critic + decentralized actors。Critic 在训练时帮助评价联合行为，actor 在比赛时不需要中央控制器。

## 9. Parameter sharing and value decomposition / 参数共享与价值分解

Homogeneous agents can share one network, often with an agent-ID input. This
reduces parameters and lets experience transfer between roles, but can erase
useful specialization if identity is absent.

同质 agent 参数共享可提高样本效率。若任务需要不同角色，可加入 agent ID、role embedding 或使用独立 heads。

Value-decomposition methods represent a team value from individual utilities:

\[
Q_{tot}=f(Q_1,\ldots,Q_N,s).
\]

VDN uses a sum; QMIX uses a monotonic mixing network. / VDN 直接相加，QMIX 用受约束 mixing network，在可分散选动作的同时利用全局训练信息。

## 10. Communication and opponent modelling / 通信与对手建模

Communication may be fixed messages or learned continuous/discrete signals.
It helps share hidden information but introduces bandwidth, protocol emergence,
and robustness questions.

Opponent modelling estimates another agent's policy, intention, or latent type.
在 Diplomacy 中，历史消息和行动可用于推断承诺是否可信；在 Boxing 中，近期动作序列可用于识别攻击习惯。模型错误也可能被对手故意利用。

## 11. Minecraft as a long-horizon agent laboratory / Minecraft 长时域智能体

Minecraft adds sparse reward, hierarchical prerequisites, partial observability,
language grounding, memory, tool use, and fuzzy goals. A stone pickaxe requires
a chain of prerequisites, so end reward gives weak feedback to early actions.

Minecraft 的价值不是“画面更复杂”，而是一个目标可能需要数千步与多个技能。纯随机探索几乎无法发现完整 crafting chain，因此系统逐渐引入 demonstrations、hierarchical planning、language-conditioned policy、skill library 和 code generation。

Detailed evolution / 详细演化见 `minecraft-agents.md`：

```text
Malmo controllable interface
 -> MineRL realistic long-horizon tasks
 -> BASALT fuzzy human-preference tasks
 -> MineDojo large multimodal knowledge
 -> VPT video pretraining
 -> STEVE-1 language conditioning
 -> Voyager code-as-action and skill memory
 -> unified multimodal and collaborative agents
```

## 12. VPT and inverse dynamics / VPT 与逆动力学

Internet video provides observations but usually no keyboard/mouse labels.
An inverse dynamics model predicts the missing action:

\[
(o_t,o_{t+1})\rightarrow\hat a_t.
\]

VPT 先在小规模有动作数据上训练 IDM，再给大量公开视频生成 pseudo-label，最后做 behavioral cloning。优点是扩大示范规模；风险是 IDM 错误成为训练标签，尤其在“画面几乎不变的无效动作”上动作不可辨识。

Forward dynamics predicts next observation from state and action:

\[
(o_t,a_t)\rightarrow\hat o_{t+1}.
\]

不要把 inverse dynamics 与 inverse reinforcement learning 混淆：前者推断动作，后者推断 reward。

## 13. Hierarchy, memory, and code as action / 层级、记忆与代码动作

A hierarchical agent turns many primitive actions into one temporally extended
skill. An LLM may plan subgoals; a controller executes; a verifier checks world
state; memory stores successful procedures and failures.

```text
goal -> plan -> retrieve/generate skill -> execute
     -> verify -> replan or store memory
```

代码作为动作提供 temporal abstraction 和可调试反馈，但生成程序可能不安全、API 可能不完整、旧 skill 可能在新环境失效。必须限制权限、验证前置条件并记录执行结果。

## 14. Fuzzy tasks, proxies, and Goodhart / 模糊任务与代理指标

“挖两块木头”有明确成功条件；“建造漂亮宫殿”没有唯一答案。BASALT 类任务可能依赖人类偏好或 VLM judge。

Proxy evaluator 让优化成为可能，也可能只奖励表面相似。专业评估应结合 pairwise human preference、任务约束、失败检查和多样性，而不是依赖单一 learned score。

## 15. Connections to Day 3 and Boxing / 与 Day 3、Boxing 的连接

- IQL/CQL practical makes non-stationarity, shared reward, and joint actions concrete.
- Imitation learning addresses sparse reward with demonstrations.
- W&B enables opponent/seed comparisons.
- Day 4 self-play uses historical/frozen opponents to reduce overfitting to random play.

第二天告诉我们，Boxing 不应只训练“打败随机对手”的策略。对手本身定义了数据分布；改进 agent 应保存 opponent 类型、独立评估 random 与 frozen/self-play opponent，并防止训练双方同时快速变化造成不稳定。

## 16. Common misconceptions / 常见误区

1. Cooperative agents need not take identical actions. / 合作目标相同不等于动作相同。
2. A Nash equilibrium need not maximize social welfare. / 纳什均衡不保证团队最优。
3. Centralized training is not automatically CTDE. / 执行仍需全局信息就不是分散执行。
4. More agents create exponential joint actions, not merely more input rows.
5. Parameter sharing does not mean observations or hidden states are shared.
6. Minecraft VLM evaluation is not the same as a policy that can act.
7. A plausible LLM plan can fail during grounded execution.

## 17. Professional glossary / 专业术语

| Term | 中文 | Meaning |
|---|---|---|
| Stochastic game | 随机博弈 | Multi-agent extension of an MDP |
| Joint action | 联合动作 | Tuple containing every agent's action |
| General-sum | 一般和 | Rewards are neither identical nor exact opposites |
| Best response | 最佳响应 | Optimal policy against fixed others |
| Nash equilibrium | 纳什均衡 | No unilateral profitable deviation |
| Non-stationarity | 非平稳性 | Data-generating dynamics change during learning |
| Credit assignment | 信用分配 | Attribute team/long-term reward to decisions |
| CTDE | 集中训练、分散执行 | Global training information, local deployed actors |
| Parameter sharing | 参数共享 | Reuse one policy network across agents |
| Opponent modelling | 对手建模 | Estimate another agent's policy or intent |
| Temporal abstraction | 时间抽象 | One high-level action spans many primitive steps |
| Language grounding | 语言落地 | Connect words to states, objects, and actions |
| Pseudo-label | 伪标签 | Label predicted by another model |
| Fuzzy task | 模糊任务 | No single exact successful output |

## 18. Self-check with answers / 自测与答案

1. **Why is IQL non-stationary? / IQL 为什么非平稳？** Other agents' policies change the effective transition/reward distribution.
2. **Why does centralized enumeration fail to scale? / 为什么难扩展？** Joint action count is \(m^N\).
3. **What is CTDE?** Use global information for training, but local information for deployed actors.
4. **IDM versus IRL? / IDM 与 IRL？** IDM infers actions from transitions; IRL infers rewards from behavior.
5. **Why use opponent pools? / 为什么使用对手池？** Reduce overfitting and forgetting against one current opponent.
