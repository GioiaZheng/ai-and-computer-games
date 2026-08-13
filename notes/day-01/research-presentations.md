# Day 1 - Course Team Research Presentations / 课程团队研究演讲

## 中文精读导读 / Detailed bilingual reading guide

四场研究报告共同讨论一个问题：**模型在训练分布上表现好，为什么到了新领域、对手、数据来源或想象环境中仍会失败？**

- Janne 研究 domain generalization、meta-learning 和 catastrophic forgetting：模型要适应新 domain，同时避免破坏旧能力。
- Mikko 研究 adversarial RL：对手不是固定噪声，而会主动寻找 policy 的弱点。
- Nima 研究 VPT 与 inverse dynamics：公开视频缺少动作标签，需要从连续画面推断 pseudo-actions。
- Ekaterina 研究 imagined GAIL：在 world model 的想象轨迹中做模仿，提高数据效率，但必须防止利用模型误差。

这些报告与 Boxing 的直接关系是：不能只训练一次、对一个 random opponent 报告平均 reward。应测试多个 opponents、seeds、checkpoint generations 和失败状态；如果使用 Alien/Minecraft 预训练表示或专家视频，还要验证 domain shift 与 action-label quality。

研究术语的直觉连接：**generalization** 问“换环境还能不能用”，**continual learning** 问“学新东西会不会忘旧东西”，**adversarial robustness** 问“对手主动攻击时是否稳定”，**imitation from observation** 问“没有动作日志能否从画面学”，**world-model imagination** 问“能否少与真实环境交互而在模型中练习”。

> 时间 / Time: Day 1, 14:30-16:00
>
> 本文件整理第一天下午的研究报告，与上午的 ML、optimization 和 RL 基础分开。

## Overview / 总览

The presentations connect four research directions:

1. **Janne Laakkonen** - domain generalization, meta-learning, and stable adaptation
2. **Mikko Turunen** - adversarial reinforcement learning and robust agents
3. **Nima Hadavi** - Video PreTraining, inverse dynamics, and ineffective actions
4. **Ekaterina Amozova** - GAIL in imagined trajectories and model-based RL

共同问题是：模型离开标准训练分布以后，怎样继续可靠地学习和行动？

---

## 1. Janne Laakkonen - Domain Generalization and Meta-Learning

### Research direction / 研究方向

Janne's work studies **domain generalization（域泛化）** and **meta-learning（元学习）**, including applications to speech deepfake detection.

- **Domain（域）**: a data-generating setting, such as a recording device, language, environment, or attack type.
- **Domain shift（域偏移）**: the test domain differs from the training domain.
- **Domain adaptation（域适应）**: use some information from the new domain to adapt the model.
- **Domain generalization（域泛化）**: perform well on an unseen domain without training on that target domain.
- **Meta-learning（元学习）**: learn a training or adaptation procedure, often summarized as “learning to learn”.

Research evolution presented:

| Stage | Focus | 中文理解 |
|---|---|---|
| Past | Few-shot domain adaptation | 用少量新域样本适应 |
| Present | Zero-shot domain generalization | 不看新域样本也要泛化 |
| Future | Continual test-time adaptation | 部署过程中持续适应 |

### Few-shot speech deepfake adaptation / 少样本语音深伪适应

The task is to detect spoofed or synthetic speech when evaluation attacks differ from known training attacks.

Relevant terms:

- **Few-shot learning（少样本学习）**: learn or adapt using only a small number of examples.
- **EER, Equal Error Rate（等错误率）**: the point where false-accept and false-reject rates are equal; lower is better.
- **Self-supervised learning, SSL（自监督学习）**: learn representations from unlabeled data using automatically constructed objectives.
- **Wav2Vec2**: a self-supervised speech representation model.
- **AASIST**: an anti-spoofing architecture for detecting fake speech.
- **ProtoNet / ProtoMAML**: meta-learning approaches for rapid adaptation.

Lecture-reported result:

| Setting | Reported result |
|---|---|
| 96-shot adaptation | In-the-Wild EER improved from 21.67% to 10.42% |
| Trainable scale | Approximately 0.14% of full fine-tuning parameters |

This is a reported slide result, not a general guarantee for every dataset.

### Meta-learned LoRA / 元学习 LoRA

**LoRA (Low-Rank Adaptation，低秩适应)** inserts small trainable matrices into a frozen pretrained model. It reduces the number of parameters that must be updated.

The presented idea combines:

```text
frozen Wav2Vec2
    + LoRA adapters
    + AASIST classifier
    + simulated domain shifts during meta-training
```

**MLDG (Meta-Learning for Domain Generalization)** simulates train/test domain changes during training so the update itself becomes more robust to unseen domains.

Lecture-reported result:

- average EER: 8.84% -> 5.30%
- trainable parameters: about 3.6M, roughly 1%
- lower variation across random seeds

### Catastrophic forgetting / 灾难性遗忘

**Catastrophic forgetting（灾难性遗忘）** happens when learning a new task severely damages performance on an old task.

```text
learn Task A -> good at A
learn Task B -> good at B, but A may collapse
```

- **Plasticity（可塑性）**: ability to learn new knowledge.
- **Stability（稳定性）**: ability to preserve old knowledge.
- **Stability-plasticity dilemma（稳定性-可塑性困境）**: adapting quickly without forgetting too much.

The research claim challenges the idea that larger Euclidean parameter movement always means more forgetting. Update **direction（方向）** can matter more than raw update **magnitude（大小）**.

### Fisher geometry / Fisher 几何

The lecture used **Fisher Information（费舍尔信息）** as a local geometry of how sensitive the old task is to parameter changes:

- high-Fisher directions: old predictions are sensitive; changes are risky
- low-Fisher directions: old predictions are flatter; changes may be safer

The proposed direction was summarized as:

\[
\Delta \theta \propto F_A^{-1} g_B,
\]

where \(F_A\) represents old-task sensitivity and \(g_B\) is the new-task gradient.

The lecture's “compass versus ruler” intuition:

- **Compass / geometry（指南针 / 几何）** decides which direction is safer.
- **Ruler / learning rate（尺子 / 学习率）** decides how far to move.

The presented **Drift-Waste Theorem** was a research claim: update components outside the preferred prior-Fisher direction may create old-task predictive drift without first-order benefit to the new task. It should be recorded as the speaker's research framing, not treated as a universally established textbook theorem.

---

## 2. Mikko Turunen - Adversarial Reinforcement Learning

### Core idea / 核心思想

**Adversarial Reinforcement Learning（对抗强化学习）** studies how a policy behaves when another adaptive agent deliberately searches for weaknesses.

Typical setup:

1. Train a victim policy \(\pi_v\).
2. Freeze the victim.
3. Train an adversary \(\pi_a\) against it.
4. The adversary discovers exploitable behavior.
5. Evaluate or retrain the victim using those difficult states.

The adversary acts as an adaptive stress test. A high average benchmark score does not automatically imply robustness.

### Why it matters / 为什么重要？

An ordinary test set is fixed. An adversary changes its strategy after observing what the victim does. This can expose:

- **out-of-distribution states, OOD states（分布外状态）**
- brittle decision rules / 脆弱决策规则
- unnatural but effective exploits / 看起来异常但有效的利用方式
- gaps between nominal performance and real robustness / 标准表现与真实鲁棒性的差距

Applications include games, robotics, autonomous systems, and environments containing malicious actors.

### Protected Agents / 受保护智能体

The presented project was titled:

```text
Protected Agents: Defending RL Policies Against Adversarial and Human Exploitation
```

The slide said the work was submitted to AAAI 2027.

Research questions:

- Do human players and PPO adversaries find the same exploits?
- Can specialized training improve policy robustness?
- Does perceived robustness agree with quantitative robustness?

Experiments mentioned Atari Boxing and Combat: Tank. Human evaluation also asked whether an agent was enjoyable, exploitable, human-like, and skilled.

Important finding: a more robust agent is not necessarily a more enjoyable opponent. Robustness and player experience are different evaluation dimensions.

### Latent-state analysis / 潜在状态分析

The blackboard pipeline was:

```text
state or observation s -> CNN encoder -> latent representation z -> action a
```

**Latent representation（潜在表示）** is the internal feature vector produced by the network. Visualizing these vectors can show whether human or adversarial play visits distinct state clusters.

This matters because aggregate reward may say that a failure occurred, while representation-space analysis can help explain *where* and *how* the policy entered an unfamiliar regime.

---

## 3. Nima Hadavi - Video PreTraining and Inverse Dynamics

### Research question / 研究问题

The talk title was:

```text
The effect of ineffective actions on inverse dynamics models
```

The initial application was learning to play Untitled Goose Game through **Video PreTraining (VPT，视频预训练)**.

VPT is useful when pure RL faces **sparse reward（稀疏奖励）**. Random exploration may rarely discover meaningful behavior, while large video collections already contain useful demonstrations.

### Behavioral cloning and missing actions / 行为克隆与缺失动作标签

Behavioral cloning normally needs expert pairs:

\[
D_E = \{(o_n,a_n)\}_{n=1}^{N}.
\]

Internet gameplay video supplies observations \(o_n\), but usually does not record keyboard or controller actions \(a_n\). Therefore, it cannot directly train an action classifier.

### Inverse versus forward dynamics / 逆动力学与正向动力学

**Inverse Dynamics Model (IDM，逆动力学模型)**:

\[
(o_t,o_{t+1}) \rightarrow a_t.
\]

It predicts the action that caused the observed transition.

**Forward Dynamics Model (FDM，正向动力学模型)**:

\[
(o_t,a_t) \rightarrow o_{t+1}.
\]

It predicts the next observation after an action.

VPT uses an IDM to infer **pseudo-labels（伪标签）** for unlabeled video. The inferred state-action data can then train a behavioral-cloning policy.

### Large-scale VPT pipeline / 大规模 VPT 流程

The lecture described the following scale:

```text
about 2,000 hours of action-labelled contractor data
    -> train inverse dynamics model

about 270,000 hours of raw Internet video
    -> filter to about 70,000 useful hours
    -> infer actions with the IDM
    -> behavioral cloning
    -> pretrained policy
```

The main risk is **pseudo-label error propagation（伪标签误差传播）**: systematic IDM mistakes become labels for the policy.

### Ineffective actions / 无效动作

An **ineffective action（无效动作）** is recorded but causes no visible state change.

```text
interaction action -> no visible change
NOOP               -> no visible change
```

- **Action effect（动作效果）**: a change caused by the player's action.
- **Exogenous effect（外生效果）**: a change caused by something else in the environment.
- **NOOP (no operation，无操作)**: an action representing “do nothing”.

If two different actions produce nearly identical observation transitions, the IDM cannot reliably infer which one was pressed. This is an **identifiability problem（可辨识性问题）**.

### Controlled experiment / 对照实验

```text
raw demonstrations -> Raw IDM

raw demonstrations
    -> detect ineffective actions
    -> relabel as NOOP
    -> Clean IDM
```

The lecture reported that the clean-trained model outperformed the raw-trained model in the shown Untitled Goose Game and Craftax experiments. The cautious conclusion is that ambiguous ineffective-action labels can degrade inverse-dynamics learning; the result should not be generalized beyond the presented experimental conditions without the full study.

---

## 4. Ekaterina Amozova - GAIL in Agent's Imagination

### Research direction / 研究方向

The talk title was:

```text
GAIL in agent's imagination
```

The work combines:

- **Generative Adversarial Imitation Learning (GAIL，生成对抗模仿学习)**
- **model-based reinforcement learning（基于模型的强化学习）**
- Dreamer-style **imagined rollouts（想象轨迹）**

The key question is whether imitation learning can use trajectories generated inside a learned world model instead of depending only on expensive real-environment interaction.

### RL versus imitation learning / RL 与模仿学习

- RL learns primarily from environment reward.
- Imitation learning learns from expert demonstrations.
- GAIL trains a discriminator to distinguish expert transitions from learner transitions.
- The policy receives a learned imitation signal for making its behavior look expert-like.

### DreamerV3 / DreamerV3 结构

**DreamerV3** is a model-based RL architecture with:

| Component | Function / 作用 |
|---|---|
| World model / 世界模型 | Learns environment dynamics from real experience |
| Actor / 行动者 | Chooses actions |
| Critic / 评价者 | Estimates long-term value |

The world model learns from real interaction, while the actor and critic can learn from trajectories imagined by the world model.

### Dreamer plus discriminator / Dreamer 加判别器

Conceptual loop:

```text
real experience -> world model
world model -> imagined trajectory
imagined trajectory + human trajectory -> discriminator
discriminator -> imitation reward
imitation reward -> actor and critic update
```

Potential benefit: more policy updates per unit of real environment experience.

Main risks:

- model errors accumulate in imagined trajectories
- discriminator and policy objectives can destabilize each other
- the policy may exploit world-model errors
- reward scale and sign may produce unintended behavior

The lecture noted that an always-positive reward can encourage longer survival, while a reward that becomes negative may make episode termination attractive. Even learned rewards require incentive analysis.

Preliminary Atari environments mentioned were Seaquest and Alien, used as simpler staging environments before harder tasks.

Other research directions briefly mentioned:

- DISCO: time-series interaction prediction between cell types
- Reward Machines plus imitation learning

A **Reward Machine（奖励机器）** represents reward logic as a finite-state machine, making multi-step task structure explicit.

---

## 5. Connections between the four talks / 四个报告之间的联系

| Research problem | Failure being addressed | Main idea |
|---|---|---|
| Domain generalization | New test domains | Learn representations or updates that transfer |
| Continual adaptation | Catastrophic forgetting | Respect old-task geometry while learning new tasks |
| Adversarial RL | Adaptive exploitation | Train an opponent to discover policy weaknesses |
| Video PreTraining | Sparse reward and missing action labels | Infer actions from observation transitions |
| Ineffective actions | Ambiguous supervision | Clean or relabel transitions with no visible effect |
| GAIL in imagination | Expensive environment interaction | Learn imitation policies from world-model rollouts |

Shared theme:

```text
high average training performance is not enough
    -> inspect domain shift
    -> inspect adversarial pressure
    -> inspect label ambiguity
    -> inspect model-generated experience
```

## 6. Boxing project relevance / 与 Boxing 项目的关系

- Mikko's work is directly relevant because Atari Boxing can contain exploitable policy behavior.
- Latent-state visualization can compare states visited by random, trained, human, and adversarial policies.
- Nima's work matters if human gameplay video is used to create demonstrations without action logs.
- Ekaterina's work connects to learning an imitation reward and reducing real rollout cost.
- Janne's work motivates testing whether a policy generalizes across opponents, seeds, wrappers, or modified game dynamics.

Practical lesson: do not evaluate the Boxing agent only by one mean reward. Record several seeds, multiple opponents, score difference, failure states, and recovery after unfamiliar situations.

## Source note / 来源说明

This file reorganizes the Day 1 research-presentation material previously stored inside the older combined course note. Numerical results and named research claims are explicitly treated as lecture-reported material. Explanations and Boxing examples are learning supplements.
