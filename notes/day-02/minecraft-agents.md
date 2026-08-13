# Day 2 - Reinforcement Learning and Minecraft / 强化学习与 Minecraft

> Lecturer / 讲师: Anssi Kanervisto
>
> Time / 时间: Day 2, 11:00-12:00
>
> This note combines the live classroom notes with the reading links sent by Anssi after the lecture.

## 中文精读导读 / Detailed bilingual reading guide

Minecraft 演讲展示了 agent action 的层级演化：最底层是键盘鼠标动作，中间层是“挖木头、合成工具”等 reusable skill，最高层是自然语言 subgoal 或生成程序。层级越高，规划 horizon 越短，但每层接口都可能带来 grounding、执行与验证错误。

VPT 解决“公开视频有画面却没有按键”的问题：先训练 inverse dynamics model 从 \((o_t,o_{t+1})\) 推断 \(a_t\)，再为大规模视频生成 pseudo-label，最后做 behavioral cloning。STEVE-1 再加入 language/goal conditioning；Voyager 把生成代码当成 temporally extended action，并把成功代码存入 skill library。

初学者应把系统拆成四个问题：**perception 看到了什么，planning 下一步目标是什么，control 如何执行，verification 是否真的成功。** LLM 能生成语义合理的计划，不代表世界中有资源，也不代表 controller 能完成动作，所以 memory 与 replanning 必须建立在环境反馈上。

模糊任务如“建造漂亮宫殿”没有唯一正确 state，常用人类偏好或 VLM proxy 评价。Proxy 可扩展评估，也会带来 Goodhart/reward hacking：模型可能优化“看起来像”而不满足真正功能。

## 1. Why Minecraft? / 为什么使用 Minecraft？

Minecraft is a useful **embodied-agent environment（具身智能体环境）** because an agent must connect perception, decisions, actions, and long-term consequences inside one persistent world.

Compared with a short Atari task, Minecraft adds:

- **long horizon（长时域）**: success may require thousands of actions
- **sparse reward（稀疏奖励）**: useful feedback can arrive very late
- **hierarchical dependencies（层级依赖）**: later items require earlier tools and resources
- **open-endedness（开放性）**: many goals have multiple valid solutions
- **language grounding（语言落地）**: instructions must be connected to visible objects and actions
- **partial observability（部分可观测性）**: the agent sees only its current view and limited state
- **human collaboration（人机协作）**: humans can give vague, changing, or conversational instructions

Example dependency chain:

```text
wood
  -> planks
  -> crafting table
  -> sticks
  -> wooden pickaxe
  -> cobblestone
  -> stone pickaxe
```

If the final goal is “obtain a stone pickaxe”, reward at the end does not directly explain which earlier action failed. This is a **credit-assignment problem（信用分配问题）**.

## 2. Evolution of Minecraft environments / Minecraft 环境演化

### 2.1 Project Malmo / Project Malmo

Project Malmo was an early platform for Minecraft research.

- exposed a programmable interface for agents
- supported Python-based experiments
- enabled tasks such as navigation and maze solving
- established Minecraft as a controlled research environment

Beginner intuition: first make it possible for a research agent to enter, observe, and control Minecraft reliably.

### 2.2 MineRL / MineRL

MineRL moved toward more realistic Minecraft tasks and long item-dependency chains.

Why pure RL struggled:

1. Random exploration rarely discovers the entire crafting chain.
2. Rewards near the final objective provide weak guidance for early actions.
3. The action and observation spaces are large.
4. A failed prerequisite can invalidate a long trajectory.

Strong approaches therefore used **hierarchy（层级结构）**, **task decomposition（任务分解）**, and **imitation learning（模仿学习）** rather than relying only on end-to-end RL from scratch.

### 2.3 CraftAssist / CraftAssist

CraftAssist studied human-agent collaboration through chat and building.

Important concepts:

- **instruction following（指令遵循）**
- **dialogue grounding（对话落地）**
- **reference resolution（指代消解）**: understanding what “that block” refers to
- **collaborative construction（协作建造）**

The task is not only to optimize a score. The agent must understand a human's intended result and respond appropriately during interaction.

### 2.4 MineRL BASALT / MineRL BASALT

BASALT introduced **fuzzy tasks（模糊任务）**, for example “build a waterfall”.

These tasks have no single exact target state and no simple binary success test. Several outputs may be valid but differ in quality.

Evaluation may therefore use:

- human preferences
- pairwise comparisons
- learned visual or language-based judges
- task-specific proxy metrics

Risk: a proxy evaluator may reward something that looks superficially correct without satisfying the human's real intention. This connects to **Goodhart's Law（古德哈特定律）**: when a measure becomes a target, it may stop being a good measure.

### 2.5 MineDojo and MineCLIP / MineDojo 与 MineCLIP

MineDojo expanded the scale of Minecraft tasks and knowledge sources.

- thousands of tasks
- knowledge from videos, wiki-style resources, and Internet text
- MineCLIP-style text-image alignment

**Semantic alignment（语义对齐）** means that visual observations and language goals are represented so that matching scenes and descriptions become close in representation space.

Evolution summary:

```text
Malmo: provide a controllable research interface
  -> MineRL: realistic long-horizon survival and crafting
  -> CraftAssist: language and human collaboration
  -> BASALT: fuzzy goals and preference-based evaluation
  -> MineDojo: large-scale tasks, knowledge, and multimodal alignment
```

## 3. Video PreTraining / 视频预训练

### 3.1 The missing-action problem / 缺少动作标签的问题

Internet gameplay video contains observations but usually not the player's keyboard and mouse actions. Standard behavioral cloning needs pairs:

\[
D_E = \{(o_t,a_t)\}.
\]

An **Inverse Dynamics Model (IDM，逆动力学模型)** predicts the action from consecutive observations:

\[
(o_t,o_{t+1})\rightarrow \hat a_t.
\]

VPT uses action-labelled gameplay to train an IDM, applies it to large unlabeled video collections, and then behaviorally clones a policy from the pseudo-labelled data.

```text
small action-labelled dataset
  -> train IDM
large unlabeled video dataset
  -> infer actions
  -> pseudo-labelled demonstrations
  -> behavioral cloning
  -> pretrained Minecraft policy
```

### 3.2 What VPT contributes / VPT 的贡献

- scales imitation learning using Internet video
- gives the policy broad low-level gameplay behavior
- provides a foundation that can later be fine-tuned

Main limitations:

- IDM errors become policy-training labels
- video quality and filtering affect the learned behavior
- imitation reproduces the demonstration distribution but does not automatically solve new language goals

## 4. Language-conditioned agents / 语言条件智能体

### STEVE-1

STEVE-1 combines VPT-style behavioral knowledge with MineCLIP-style language and visual goal conditioning.

Conceptual shift:

```text
VPT: knows how Minecraft behavior looks
  -> STEVE-1: selects behavior according to a language goal
```

**Goal conditioning（目标条件化）** means the policy receives the desired goal as part of its input, so the same policy can act differently for different instructions.

## 5. LLM planning and structured skills / LLM 规划与结构化技能

### 5.1 Ghost in the Minecraft / Ghost in the Minecraft

The classroom framing uses a high-level planner plus coded low-level behaviors:

```text
language goal
  -> LLM planner
  -> structured skill such as explore, mine, craft, or dig
  -> low-level controller
```

Advantage: the planner selects from a smaller, meaningful action set.

Limitation: the agent cannot exceed the manually provided skill interface easily.

### 5.2 Plan decomposition / 计划分解

For “obtain a stone pickaxe”, an LLM can produce:

```text
collect wood
  -> craft planks
  -> craft table and sticks
  -> craft wooden pickaxe
  -> mine cobblestone
  -> craft stone pickaxe
```

LLMs are useful because they contain semantic and procedural knowledge. However, a textually plausible plan may fail because:

- a prerequisite is missing
- the recipe is hallucinated
- the required object is unavailable nearby
- the current world state contradicts the plan
- execution fails even though the plan is logically correct

This is why a practical agent needs **verification（验证）** and **replanning（重新规划）**.

## 6. Voyager and code as action / Voyager 与“代码即动作”

Voyager-style agents let the model generate executable code, commonly using a Minecraft programming interface.

```text
new task
  -> generate program
  -> execute program
  -> observe error or success
  -> revise program
  -> store successful skill
```

Here, a program is a high-level action containing many primitive mouse and keyboard actions. This is **temporal abstraction（时间抽象）**: one decision controls behavior over a longer interval.

Benefits:

- execution errors provide concrete feedback
- programs are inspectable and reusable
- one skill can solve a recurring subtask
- skill composition supports longer tasks

Risks:

- generated code may be invalid or unsafe
- APIs may not expose every useful behavior
- a stored skill can fail when world conditions change

## 7. Memory and skill reuse / 记忆与技能复用

| Memory type | 中文 | Minecraft example |
|---|---|---|
| Declarative memory | 陈述性记忆，知道“是什么” | A stone pickaxe requires sticks and cobblestone |
| Procedural memory | 程序性记忆，知道“怎么做” | Executable routine for collecting wood and crafting tools |
| Episodic memory | 情景记忆，记住过去经历 | A previous attempt failed because no tree was nearby |

A **skill library（技能库）** stores successful routines such as:

- collect wood
- craft a table
- use a furnace
- make a shield
- fight a zombie

With retrieval, the agent can search for a previously solved similar subtask instead of generating every behavior from zero.

## 8. Modular versus unified agents / 模块化与统一模型

### Modular design / 模块化设计

```text
planner -> memory -> skill selection -> controller -> verifier
```

Advantages: components are easier to inspect, replace, and debug.

Disadvantages: interfaces are hand-designed and errors can propagate between modules.

### Unified multimodal policy / 统一多模态策略

OmniJARVIS-style systems aim for:

```text
instruction + visual observation + interaction history
  -> multimodal policy
  -> action
```

The model must jointly connect language, perception, memory, and control. This reduces manual interfaces but creates a more difficult learning problem.

The lecture link list also included Plan4MC, JARVIS-1, OmniJARVIS, DEPS, MP5, and Pan-1. Their names and references were provided as a reading map; detailed architectures were not fully covered in the available live notes.

## 9. VLM and VLA / 视觉语言模型与视觉语言动作模型

### VLM

**Vision-Language Model (VLM，视觉语言模型)**:

```text
image or video + text -> description, answer, comparison, or score
```

It can judge whether a Minecraft building resembles “a medieval fortress”, but it does not necessarily control the game.

### VLA

**Vision-Language-Action model (VLA，视觉语言动作模型)**:

```text
visual observation + language instruction -> environment action
```

- Vision: what the agent sees
- Language: what the human requests
- Action: how the agent changes the world

In short: a VLM interprets or evaluates; a VLA acts.

## 10. From explicit tasks to creative goals / 从明确任务到创造性目标

Explicit task:

```text
mine two logs
```

Constrained task:

```text
dig a block of sand near water at night with a wooden shovel
```

This combines object, location, time, tool, and prerequisite constraints.

Creative task:

```text
build a sandstone palace with intricate details and towering minarets
```

Creative tasks are difficult because there is no unique correct trajectory or final state. Evaluation may depend on human preferences or a VLM proxy, creating opportunities for **reward hacking（奖励投机）**.

Creative Agents and Luban were included in Anssi's reading list as systems for fuzzy Minecraft tasks.

## 11. Minecraft as a general-agent benchmark / Minecraft 作为通用智能体基准

The lecture's framing was not that Minecraft equals general intelligence. Rather, Minecraft can test a broad collection of capabilities that many human players combine naturally:

- perception
- world knowledge
- planning
- long-term memory
- tool use
- efficient execution
- failure recovery
- communication and collaboration

Example benchmark tasks:

- defeat the Ender Dragon within 24 hours
- defeat it without dying
- obtain an Elytra
- collect 1,000 gunpowder within 24 hours

The gunpowder task may require understanding mechanics, designing a farm, collecting prerequisites, managing time, and recovering from execution failures.

Fair comparison requires comparable resources and interfaces: visual input, ordinary controls, and similar access to Internet or wiki knowledge.

## 12. Human-agent collaboration / 人机协作

The preferred future direction emphasized connecting human players with agents.

Research questions:

- How do humans naturally request help?
- Can the agent infer incomplete human intentions?
- Can it ask useful clarification questions?
- Can it adapt when the human changes plans?
- Does it contribute without taking control away from the human?

Minecraft therefore evaluates not only autonomous competence, but **collaborative embodied agency（协作式具身智能）**.

## 13. Evolution summary / 演化总结

```text
programmable environment
  -> realistic long-horizon benchmark
  -> imitation from large-scale video
  -> language-conditioned behavior
  -> LLM planning with structured skills
  -> code generation as action
  -> reusable skill libraries and memory
  -> unified multimodal policies
  -> fuzzy creative goals
  -> human-agent collaboration
```

This evolution changes the meaning of an action:

```text
primitive key press
  -> named skill
  -> subgoal
  -> tool call
  -> generated program
```

The higher-level action improves long-horizon planning, but every abstraction layer can introduce new planning, execution, and verification errors.

## 14. Anssi's reading links / Anssi 提供的阅读链接

### Minecraft environments

- [Project Malmo](https://www.microsoft.com/en-us/research/project/project-malmo/)
- [MineRL paper](https://arxiv.org/abs/1907.13440) and [MineRL repository](https://github.com/minerllabs/minerl)
- [CraftAssist](https://arxiv.org/abs/1907.08584)
- MineRL BASALT: [paper 1](https://arxiv.org/abs/2312.02405), [paper 2](https://arxiv.org/abs/2303.13512)
- [MineDojo](https://minedojo.org/)

### Minecraft agents

- [OpenAI VPT](https://github.com/openai/Video-Pre-Training)
- [STEVE-1](https://sites.google.com/view/steve-1)
- [Ghost in the Minecraft](https://arxiv.org/abs/2305.17144)
- [Voyager](https://arxiv.org/abs/2305.16291)
- [Plan4MC](https://arxiv.org/abs/2303.16563)
- [JARVIS-1](https://arxiv.org/abs/2311.05997)
- [OmniJARVIS](https://arxiv.org/abs/2407.00114)
- [DEPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/6b8dfb8c0c12e6fafc6c256cb08a5ca7-Paper-Conference.pdf)
- [MP5](https://arxiv.org/abs/2312.07472v2)
- [Pan-1](https://pantograph.com/journal/pan-1)
- [Creative Agents](https://arxiv.org/abs/2312.02519)
- [Luban](https://arxiv.org/abs/2405.15414)

## 15. Quick glossary / 术语速查

| Term | 中文 | Meaning |
|---|---|---|
| Embodied agent | 具身智能体 | An agent that perceives and acts in an environment |
| Long horizon | 长时域 | Many dependent steps before success |
| Sparse reward | 稀疏奖励 | Useful rewards occur rarely |
| Hierarchical planning | 层级规划 | Break a goal into subgoals and skills |
| Language grounding | 语言落地 | Connect words to objects, states, and actions |
| Goal conditioning | 目标条件化 | Policy behavior depends on an input goal |
| IDM | 逆动力学模型 | Predict action from consecutive observations |
| Pseudo-label | 伪标签 | A label inferred by another model |
| Skill library | 技能库 | Collection of reusable procedures |
| Code as action | 代码即动作 | Use a generated program as a high-level action |
| VLM | 视觉语言模型 | Understands or evaluates visual and textual input |
| VLA | 视觉语言动作模型 | Maps visual and language input to actions |
| Fuzzy task | 模糊任务 | Task without one exact success state |

## 16. Self-check / 自测题

1. Why is obtaining a diamond difficult for pure RL?
2. What changed from Malmo to MineRL, BASALT, and MineDojo?
3. Why does VPT need an inverse dynamics model?
4. What is the difference between VPT and STEVE-1?
5. Why does code generation provide temporal abstraction?
6. Distinguish declarative, procedural, and episodic memory.
7. Why can a textually correct LLM plan still fail in Minecraft?
8. What is the difference between a VLM and a VLA?
9. Why are creative Minecraft tasks vulnerable to proxy-reward problems?
10. Design a plan-memory-verification loop for obtaining 1,000 gunpowder.

## Source boundary / 来源边界

The lecture sequence and interpretations come from the Day 2 classroom notes. The links in Section 14 are the reading list sent by Anssi after the lecture. Systems that were only named are kept as references and are not presented as if their full architectures had been explained in class.
