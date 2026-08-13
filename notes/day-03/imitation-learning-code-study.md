# Day 3 - Imitation Learning Code Study / 模仿学习代码研究

> Day 3 master guide / Day 3 总讲解：`README.md`

## 中文精读导读 / Detailed bilingual reading guide

这份笔记的重点不是证明老师的示例“能直接跑完”，而是研究公式如何落到真实张量，以及哪些实现细节会悄悄改变算法语义。

### 数据必须先于模型 / Data before model

模仿学习数据不能只保存无序图片。正确的专家记录至少应包含：

```text
episode_id, timestep, observation, action, reward, terminated, truncated
```

`episode_id` 与 `timestep` 保留轨迹顺序；`action` 必须属于同一时刻的 observation；`terminated` 与 `truncated` 决定轨迹何时结束。若这些字段缺失，BC 也许还能做图片分类，但 IRL 的 feature expectation 与 GAIL 的 trajectory/occupancy 语义会变得不可靠。

### 三份代码分别学什么？ / What does each script learn?

- `bc_agent.py` 学 \(\pi_\theta(a\mid s)\)：给一帧图像预测专家动作。
- `irl_agent.py` 学 \(w\)：给 encoder feature 加权，形成 \(r_w(s)=w^\top\phi(s)\)，再用 PPO 学 policy。
- `gail_agent.py` 学 discriminator \(D(s,a)\) 和 policy：判别器产生 imitation reward，PPO 根据该 reward 更新。
- `utils.py` 负责预处理、rollout、wrapper 和 evaluation；utility code 出错同样会让算法失真。

### 为什么是 512 和 530？ / Why 512 and 530?

BC CNN 把 `(B,1,64,64)` 图像编码成 `(B,512)` feature。Alien 有 18 个离散动作，one-hot 后是 `(B,18)`。GAIL 把二者沿 feature 维拼接：

\[
512+18=530.
\]

所以 discriminator 输入是 `(B,530)`，输出通常是 `(B,1)`。如果动作没有 one-hot、batch 维拼错或 action count 写死后换了环境，代码可能报 shape error，也可能更危险地以错误语义继续运行。

### 最重要的实现风险 / Most important implementation risks

1. **State-action pairing：** 训练 discriminator 与生成 reward 必须都使用 \((s_t,a_t)\)，不能一处使用动作前状态、一处使用动作后状态。
2. **Trajectory discounting：** IRL feature expectation 应按 episode 分组并使用 \(\gamma^t\)，不能只对所有 frame 做普通平均。
3. **GAIL sampling：** expert state 必须保留它自己的 expert action；随机重排后不能再用另一条样本的动作。
4. **Numerical stability：** 若 discriminator 返回 logits，应使用 `BCEWithLogitsLoss`；若先 sigmoid，再对概率做 BCELoss，二者不能重复。
5. **Evaluation boundary：** “模型成功加载”只证明结构兼容；“action accuracy 高”只证明拟合示范；真正游戏能力仍需 environment return、胜率与多 seed 测试。

这些代码最适合作为教学骨架。可复用的是网络结构、预处理思路和交替训练框架；在项目中使用前必须先修复数据契约、设备一致性、轨迹语义和评估协议。

## 0. Source status / 材料状态

This note studies the files supplied with the 2026 UEF imitation-learning
lecture. The originals are preserved unchanged under
`materials/day-03/imitation-learning-2026/` and are excluded from Git.

| Source | Role / 作用 | Status / 状态 |
|---|---|---|
| `UEF Imitation Learning.pdf` | 18-page 2026 lecture, Ekaterina Amozova | Read completely / 已完整阅读 |
| `bc_agent.py` | Behavioral Cloning (BC) example | Syntax checked / 已做语法检查 |
| `irl_agent.py` | Feature-expectation IRL example | Syntax checked / 已做语法检查 |
| `gail_agent.py` | GAIL example | Syntax checked / 已做语法检查 |
| `utils.py` | Shared data, rollout, wrapper, and evaluation helpers | Syntax checked / 已做语法检查 |
| `bc_model_example.pt` | Pretrained BC model state dictionary | Keys and tensor shapes inspected / 已检查权重结构 |
| expert demonstrations | Atari Alien observations and actions | **Not supplied because the file is too large** |

The four Python files compile, but a full training run was deliberately not
attempted. The expert dataset is missing, `stable-baselines3` is not installed
in the current `pettingzoo` environment, Atari ROM setup can be fragile, and
the teacher explicitly recommended not spending class time on first-time Atari
setup.

Current relevant environment snapshot:

```text
Python: 3.12.13
PyTorch: 2.13.0+cu126
Gymnasium: 1.3.0
ALE-Py: 0.12.0
Stable-Baselines3: not currently installed
```

## 1. Big picture / 总体结构

All three methods use the same Atari Alien task, but they learn different
objects:

```text
expert frames + expert actions
            |
            +-> BC: learn policy logits directly
            |
            +-> IRL: reuse BC encoder -> learn reward weights -> train PPO
            |
            `-> GAIL: reuse BC encoder -> train discriminator -> train PPO
```

In mathematical shorthand:

\[
\text{BC:}\quad
D_E=\{(s_i,a_i)\}_{i=1}^{N}
\longrightarrow \pi_\theta(a\mid s),
\]

\[
\text{IRL:}\quad
D_E\longrightarrow \mu_E
\longrightarrow w
\longrightarrow r_w(s)=w^\top\phi(s)
\longrightarrow \pi,
\]

\[
\text{GAIL:}\quad
(s,a)_E,(s,a)_\pi
\longrightarrow D_\omega(s,a)
\longrightarrow \hat r(s,a)
\longrightarrow \pi.
\]

`BC` is an offline supervised-learning example. `IRL` and `GAIL` use PPO and
must repeatedly interact with the environment.

## 2. Shared data contract and preprocessing / 数据格式与预处理

### 2.1 Expert dataset / 专家数据集

`utils.load_expert_dataset(x)` scans the local `expert_trajectories/` directory
for filenames beginning with `x`. Each matching `.npz` file is expected to
contain:

```python
data["image"]   # Atari observation frames
data["action"]  # actions aligned with those frames
```

The loader uses `image[:-1]` but all actions. This assumes a standard
transition archive containing one more observation than action:

\[
(s_0,a_0,s_1,a_1,\ldots,a_{T-1},s_T).
\]

After dropping the final state, the desired alignment is

\[
(s_0,a_0),(s_1,a_1),\ldots,(s_{T-1},a_{T-1}).
\]

Because the demos are unavailable, we cannot verify filenames, array keys,
dtypes, action encoding, frame shape, or whether episode boundaries were
preserved. Those are part of the experiment specification, not minor details.

### 2.2 Image preprocessing / 图像预处理

`utils.preprocess` performs three operations:

1. RGB to grayscale using

\[
Y=0.2989R+0.5870G+0.1140B.
\]

2. Divide by 255 when values exceed 1, giving approximately \([0,1]\).
3. Convert the result to \(64\times64\).

Important implementation boundary / 重要实现边界：the third step uses
`np.resize`, which is **array reshaping with repetition or truncation**, not
image resampling. It can distort spatial structure. A reproducible experiment
should use an actual image resize operation such as OpenCV, Pillow, or a
TorchVision transform and record its interpolation rule.

## 3. Pretrained model inspection / 预训练模型检查

`bc_model_example.pt` is not encoder-only. It contains the complete BC state
dictionary: convolutional encoder, 512-unit hidden layer, and 18-action head.

### 3.1 Tensor shapes / 张量形状

For a batch of grayscale frames:

\[
x\in\mathbb R^{B\times1\times64\times64}.
\]

The spatial output formula for a convolution is

\[
H_{out}
=\left\lfloor\frac{H_{in}+2p-k}{s}\right\rfloor+1.
\]

Applying the supplied layers gives:

| Stage | Operation | Output shape |
|---|---|---|
| Input | grayscale frame | \(B\times1\times64\times64\) |
| Conv 1 | 32 filters, \(8\times8\), stride 4 | \(B\times32\times15\times15\) |
| Conv 2 | 64 filters, \(4\times4\), stride 2 | \(B\times64\times6\times6\) |
| Conv 3 | 64 filters, \(3\times3\), stride 1 | \(B\times64\times4\times4\) |
| Flatten | \(64\cdot4\cdot4\) | \(B\times1024\) |
| Hidden | linear + ReLU | \(B\times512\) |
| BC head | linear | \(B\times18\) logits |

The state dictionary has 10 tensors and **605,874 trainable scalar
parameters**. The keys are:

```text
conv.0.weight, conv.0.bias
conv.2.weight, conv.2.bias
conv.4.weight, conv.4.bias
fc.1.weight, fc.1.bias
fc.3.weight, fc.3.bias
```

IRL and GAIL remove the final `fc.3` action head and reuse the earlier weights
as an encoder. Therefore their learned feature vector is

\[
\phi(s)\in\mathbb R^{512}.
\]

## 4. Behavioral Cloning walkthrough / BC 代码逐步解释

### 4.1 What the model learns / 模型学什么

`BCModel` maps a preprocessed frame to 18 action logits:

\[
z_\theta(s)\in\mathbb R^{18},
\qquad
\pi_\theta(a\mid s)
=\frac{e^{z_a}}{\sum_{j=1}^{18}e^{z_j}}.
\]

The greedy evaluation action is

\[
\hat a=\arg\max_a z_\theta(s)_a.
\]

### 4.2 Loss / 损失函数

For an expert class index \(a_E\), `nn.CrossEntropyLoss` implements

\[
\mathcal L_{BC}
=-\frac{1}{B}\sum_{i=1}^{B}
\log \pi_\theta(a_{E,i}\mid s_i).
\]

Training follows the lecture exactly:

```text
load (state, action) batch
 -> preprocess frames
 -> compute 18 logits
 -> compare with expert actions
 -> backpropagate
 -> Adam update
 -> save bc_model.pt
```

### 4.3 Action-format uncertainty / 动作格式尚未确认

The slide says one-hot actions should be converted to class labels when
required, but the code passes `action` directly to `CrossEntropyLoss`.
PyTorch accepts either:

- integer class indices: shape \([B]\), dtype `long`; or
- class-probability targets: shape \([B,18]\), floating point.

Without the demos, we cannot tell which contract the archive uses. This must be
checked before treating a successful loss computation as evidence that the
labels are correct.

### 4.4 What the pretrained file permits / 现有权重能做什么

The supplied weights are sufficient to reconstruct the BC network. However,
`bc_agent.py` immediately loads the missing dataset and starts training before
the commented evaluation block. It has no `if __name__ == "__main__"` guard
and no evaluation-only CLI. Therefore simply running or importing the script
is not a clean pretrained-model evaluation path.

## 5. IRL walkthrough / IRL 代码逐步解释

### 5.1 Intended reward model / 目标奖励模型

The lecture first chooses a reward-family "skeleton":

\[
r_w(s)=w^\top\phi(s),
\qquad
w,\phi(s)\in\mathbb R^{512}.
\]

The code initializes \(w=0\), wraps Alien so PPO receives this learned reward,
and alternates between policy optimization and reward-weight updates.

### 5.2 Correct feature expectation / 正确的特征期望

For expert trajectories \(\tau_E^{(m)}\), the discounted empirical expert
feature expectation should be

\[
\hat\mu_E
=\frac{1}{M}\sum_{m=1}^{M}
\sum_{t=0}^{T_m-1}\gamma^t\phi(s_t^{(m)}).
\]

The policy estimate is

\[
\hat\mu_\pi
=\frac{1}{K}\sum_{k=1}^{K}
\sum_{t=0}^{T_k-1}\gamma^t\phi(s_t^{(k)}).
\]

The teaching update is gradient-shaped feature matching:

\[
w\leftarrow w+\alpha(\hat\mu_E-\hat\mu_\pi).
\]

If the expert visits a useful feature more often than the learner, the
corresponding component of \(w\) increases, making that feature more rewarding.

### 5.3 Nested optimization / 双层训练

The script expresses the expensive IRL structure clearly:

```text
outer iteration
 -> construct reward from current w
 -> train a fresh PPO learner for 4,000 steps
 -> collect learner trajectories
 -> estimate learner feature expectation
 -> compare learner with expert
 -> update w
```

This is why IRL is described as **RL with extra steps**: reward learning wraps
around another RL problem.

### 5.4 Difference between formula and current code / 公式与当前实现的差异

The expert loader flattens observations into one list and discards episode
boundaries. `compute_feature_expectations` then treats each individual frame as
if it were a trajectory and resets `gamma_pow` to 1 every time. Its effective
calculation is approximately

\[
\tilde\mu_E=\frac{1}{N}\sum_{i=1}^{N}\phi(s_i),
\]

not the trajectory-discounted \(\hat\mu_E\) shown above. The difference matters
because IRL is supposed to distinguish early and late visits and average over
complete demonstrations.

There is also a normalization error on the policy side: `rollouts` is a
two-element container `[observations, actions]`, so `len(rollouts)` equals 2.
The code should conceptually divide by the number of observation trajectories,
`len(rollouts[0])`, not by 2.

## 6. GAIL walkthrough / GAIL 代码逐步解释

### 6.1 Discriminator input / 判别器输入

The GAIL encoder produces 512 state features. Alien has 18 discrete actions,
represented as one-hot vectors. Concatenation gives

\[
x(s,a)
=\begin{bmatrix}\phi(s)\\\operatorname{onehot}(a)\end{bmatrix}
\in\mathbb R^{512+18}
=\mathbb R^{530}.
\]

This explains the otherwise mysterious `feature_dim=530` in `Discriminator`.
The network maps 530 inputs to 265 hidden units and then to one sigmoid output:

\[
D_\omega(s,a)\in(0,1),
\]

interpreted as the estimated probability that the sample came from expert
behavior.

### 6.2 Discriminator loss / 判别器损失

With expert label 1 and policy label 0, binary cross-entropy is

\[
\mathcal L_D
=-\frac{1}{B_E+B_\pi}
\left[
\sum_{i=1}^{B_E}\log D_\omega(s_i^E,a_i^E)
+\sum_{j=1}^{B_\pi}\log(1-D_\omega(s_j^\pi,a_j^\pi))
\right].
\]

The supplied policy reward uses the lecture's non-saturating option:

\[
\hat r(s,a)=-\log(1-D_\omega(s,a)).
\]

As \(D_\omega(s,a)\) approaches 1, the discriminator considers the behavior
more expert-like and the proxy reward increases.

### 6.3 Alternating loop / 交替训练循环

```text
collect one current-policy trajectory
 -> sample expert and policy batches
 -> encode observations
 -> concatenate features and actions
 -> train discriminator
 -> train PPO for 5,000 more steps using discriminator reward
 -> repeat
```

The policy and discriminator form a moving game. This is more expressive than
offline BC but is less stable and requires environment interaction.

### 6.4 State-action pairing bug / 状态动作配对问题

The current script independently calls `random.sample` for observations and
actions. That destroys their original pairing:

\[
(s_i,a_i)\quad\not\Rightarrow\quad
(s_{p(i)},a_{q(i)})\ \text{when }p\ne q.
\]

The discriminator may therefore learn from impossible or mislabeled
state-action combinations. A correct implementation samples one index set and
uses it for both arrays, or stores transitions as tuples and samples tuples.

### 6.5 Reward alignment / 奖励时序对齐

Policy rollouts store \((s_t,a_t)\), so the discriminator is trained on current
states. `RewardWrapperGAIL.step`, however, calls the environment first and then
builds the reward from the returned observation, producing

\[
D(\phi(s_{t+1}),a_t)
\]

instead of the training pair \(D(\phi(s_t),a_t)\). Either convention can be
designed deliberately, but training and reward generation must use the same
transition definition.

## 7. Shared implementation caveats / 共同实现注意点

These are reasons to read the files as instructional scaffolding rather than a
reproducible benchmark.

### 7.1 CPU/GPU placement / CPU 与 GPU

The encoder and discriminator are moved to `device`, but many input tensors are
created on CPU without `.to(device)`. On a CUDA machine this leads to errors
such as "expected all tensors to be on the same device". In addition,
`.numpy()` cannot be called directly on a CUDA tensor; the safe pattern is
`tensor.detach().cpu().numpy()`.

Affected paths include expert/policy batches in GAIL, feature-expectation
inputs in IRL, reward-wrapper inputs, and some NumPy/Torch weight updates.

### 7.2 Gymnasium termination / 终止与截断

Gymnasium separates:

\[
\text{done}=\text{terminated}\lor\text{truncated}.
\]

`collect_trajectories` assigns only `terminated` to `done` and ignores
`truncated`. Time-limit episodes can therefore continue until the independent
`max_steps` guard instead of ending at the environment boundary.

### 7.3 Numerical stability / 数值稳定性

The expression `-log(1-D)` becomes infinite when floating-point rounding makes
\(D=1\). A stable implementation uses clamping,

\[
-\log(\max(1-D,\varepsilon)),
\]

or trains with discriminator logits and `BCEWithLogitsLoss`.

### 7.4 Hard-coded action count / 写死的动作数

`one_hot_encode(action, 18)` assumes Alien's full 18-action space. For another
Atari game or a minimal action set, use `env.action_space.n` and ensure the BC
head, discriminator input, demonstrations, and policy all share that value.

### 7.5 Script lifecycle / 脚本生命周期

All three agent files execute dataset loading, GUI environment creation, and
training at import time. Adding a `main()` function, an
`if __name__ == "__main__"` guard, command-line options, deterministic seeds,
and separate `train`/`evaluate` modes would make experiments inspectable and
testable.

`render_mode="human"` is also expensive during training and problematic in
headless WSL sessions. Training normally uses no render mode; visualization is
reserved for a short evaluation run.

### 7.6 Environment closing / 环境关闭位置

`evaluate_model` closes the environment inside the episode loop. Closing should
happen after all episodes, otherwise later resets depend on backend-specific
behavior of an already closed environment.

## 8. What is reusable now? / 现在可以复用什么？

Useful parts that can be reused conceptually:

- the 64x64 grayscale CNN architecture and inspected pretrained BC weights;
- the BC cross-entropy training structure;
- the 512-feature encoder extraction pattern;
- the 530-dimensional GAIL discriminator design;
- the custom reward-wrapper idea for connecting learned rewards to PPO;
- the lecture's BC -> IRL -> GAIL -> AIRL conceptual progression.

Parts that must be supplied or repaired before a real run:

- expert demonstrations with documented shapes and episode boundaries;
- Atari ROM/environment installation and a headless smoke test;
- Stable-Baselines3 installation compatible with the current Gymnasium stack;
- correct paired sampling for GAIL;
- consistent tensor devices and scalar reward types;
- correct trajectory discounting and normalization for IRL;
- matching \((s_t,a_t)\) semantics between discriminator training and rewards;
- seeds, checkpoints, metrics, and a held-out evaluation protocol.

## 9. Recommended learning sequence / 建议学习顺序

1. Load `bc_model_example.pt` in a small evaluation-only script without demos.
2. Verify one preprocessed frame has shape \((1,1,64,64)\) and produces 18 logits.
3. Once demonstrations are available, inspect their keys, dtypes, shapes, and
   state-action alignment before training.
4. Train BC first and report held-out action accuracy plus environment return.
5. Repair and test the trajectory collector independently.
6. Implement GAIL with paired transitions and device-safe tensors.
7. Attempt feature-expectation IRL only after episode boundaries and discounting
   are trustworthy.

This follows both engineering risk and algorithmic complexity. BC gives the
fastest check that the observation/action contract is correct; GAIL and IRL add
environment interaction, learned rewards, and more ways for silent errors to
look like learning.

## 10. Boxing connection / 与 Boxing 项目的连接

The supplied weights were trained for **Alien**, not Boxing. The convolutional
architecture may be reusable as an initialization, but the 18-action head and
visual features are task-specific. Transfer should be treated as an experiment,
not as an assumption.

For Boxing, first define a demonstration record with explicit episode IDs:

```text
episode_id, timestep, observation, action, reward, terminated, truncated
```

Then compare methods under the same held-out seeds:

\[
\text{BC action accuracy},\quad
\text{mean return},\quad
\text{win rate},\quad
\text{score difference}.
\]

The key lesson from these files is broader than one Atari game: imitation
learning depends as much on correct trajectory semantics and data contracts as
on the neural-network formula.

## 11. Quick glossary / 术语速查

| Term | 中文 | Meaning in these files / 在代码中的含义 |
|---|---|---|
| Logit | 未归一化分类分数 | One of 18 raw BC action scores |
| Encoder | 编码器 | CNN mapping a frame to 512 learned features |
| Feature expectation | 特征期望 | Discounted average feature count across trajectories |
| Reward wrapper | 奖励包装器 | Replaces environment reward with a learned reward |
| Discriminator | 判别器 | Predicts whether a state-action sample is expert-like |
| Proxy reward | 代理奖励 | Reward computed from learned imitation similarity |
| State-action alignment | 状态动作对齐 | Preserving the expert action belonging to each state |
| Terminated | 任务终止 | Natural task-ending condition |
| Truncated | 截断 | External cutoff such as a time limit |
| State dictionary | 参数字典 | Named PyTorch tensors stored in the `.pt` file |

## Source boundary / 来源边界

Algorithm structure and formulas come from the 2026 UEF lecture and supplied
teaching scripts. Tensor-shape inspection, implementation caveats, corrected
formula-to-code comparisons, and the Boxing transfer discussion are study
analysis. The original scripts and pretrained weights remain unchanged.
