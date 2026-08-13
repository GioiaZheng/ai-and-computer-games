# Day 4 - Train the Boxing Agent / 第四天：训练 Boxing 智能体

## 0. Project objective / 项目目标

Day 4 turns the first three days into one reproducible agent-training project.
The goal is not merely to produce a checkpoint, but to establish evidence that
the implementation learns, can be resumed, and generalizes beyond one training
opponent.

第四天把课程知识真正组合起来：Day 1 提供 DQN、optimization 和 evaluation；Day 2 提供 opponent distribution、non-stationarity 与 self-play 思想；Day 3 提供 experiment tracking、代码审查和数据契约。最终产物包括可上传的训练源码、被 Git 忽略的 checkpoint/results/report，以及清楚的 source-versus-modified 解释。

## 1. Environment and agents / 环境与智能体

PettingZoo Atari Boxing is a two-player competitive environment. The training
agent is `first_0`; the other active agent is controlled by a random policy or
a periodically frozen snapshot.

Boxing 是近似 zero-sum 的 1v1 游戏：一方击中通常意味着另一方受损。训练只让 `first_0` 更新参数；对手在每个 episode 开始时固定为 random 或 snapshot，减少一局内部的 policy 非平稳性。

One transition for the learner is

\[
e_t=(s_t,a_t,r_t,s_{t+1},d_t).
\]

State \(s_t\) is a stack of four grayscale 84x84 frames. A single frame lacks
motion direction, so stacking approximates velocity information. / 状态使用 4 帧堆叠，帮助网络判断角色和拳头的运动趋势。

## 2. Baseline and limitations / Day 3 baseline 与局限

The preserved baseline trained a compact DQN only against a random opponent.
It already included uint8 replay, a CNN, target network, epsilon decay, Huber
loss, clipping, checkpoints, and CSV logging.

旧 baseline 是合理起点，但存在以下项目风险：

1. **Opponent overfitting / 对手过拟合：** 只打 random，可能学会利用随机行为，却不会应对有策略的对手。
2. **Maximization bias / 最大化偏差：** 同一个 target network 同时选择和评价下一动作。
3. **Representation coupling / 表示耦合：** 普通 head 必须从零学习状态价值和动作差异。
4. **No fixed-seed evaluation / 无固定种子评估：** training reward 不能替代独立 evaluation。
5. **Latest-only checkpoint / 只保存最新模型：** 最后模型不一定是评估最好的模型。
6. **Weak diagnostics / 诊断不足：** 只记录 loss，无法检查 Q scale、TD error 和 gradient。
7. **No resume state / 无完整恢复：** checkpoint 不含 target/opponent/best score 等 Day 4 状态。

## 3. Double DQN / 双重 DQN

The online network predicts the selected action value:

\[
Q_\theta(s_t,a_t).
\]

Plain DQN target:

\[
y_t^{DQN}=r_t+gamma(1-d_t)
\max_{a'}Q_{\theta^-}(s_{t+1},a').
\]

The `max` chooses the largest noisy estimate and evaluates it with the same
values, which can produce overestimation. Double DQN separates selection and
evaluation:

\[
a^*=\arg\max_{a'}Q_\theta(s_{t+1},a'),
\]

\[
y_t^{Double}=r_t+gamma(1-d_t)
Q_{\theta^-}(s_{t+1},a^*).
\]

中文理解：online network 负责回答“下一步选哪个动作”，target network 只回答“这个动作值多少”。这不能消除所有误差，但降低同一噪声同时负责选择和打分造成的偏高估计。

Code mapping / 代码对应：

```python
next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
next_values = target_net(next_states).gather(1, next_actions).squeeze(1)
targets = rewards + gamma * next_values * (1.0 - dones)
```

## 4. Dueling architecture / Dueling 网络

Dueling DQN decomposes Q into state value and action advantage:

\[
Q(s,a)=V(s)+A(s,a)-\frac1{|\mathcal A|}
\sum_{a'}A(s,a').
\]

\(V(s)\) asks whether the visual situation is generally good; \(A(s,a)\) asks
whether one action is better than the state's average. Centering advantage
makes the decomposition identifiable.

Boxing 中有些画面无论按哪个键都很危险或很有利，\(V\) 可以共享这种判断；动作 head 再学习出拳、移动与无动作之间的相对差异。Dueling 并不增加动作数量，也不自动解决 exploration，只改变 Q representation。

## 5. Replay, target network, and loss / 回放、目标网络与损失

Replay stores compact uint8 frame stacks on CPU. Only sampled mini-batches move
to CUDA. / 回放池若全放 GPU 会快速耗尽 GTX 1650 Ti 显存，因此只搬运当前 batch。

The target network is copied from the online network every fixed number of
steps. It provides a slowly moving bootstrap target. Huber loss is

\[
\operatorname{Huber}(\delta)=
\begin{cases}
\frac12\delta^2,&|\delta|\le1,\\
|\delta|-\frac12,&|\delta|>1,
\end{cases}
\qquad
\delta=y-Q_\theta(s,a).
\]

大 TD error 时 Huber 线性增长，比 MSE 更不容易产生极端梯度；gradient clipping 再限制整组参数梯度 norm。两者都是稳定机制，不是提高最终分数的保证。

## 6. Exploration schedule / 探索计划

\[
\epsilon_t=
\max\left(\epsilon_{end},
\epsilon_{start}+
\frac{t}{T_{decay}}
(\epsilon_{end}-\epsilon_{start})
\right).
\]

Training begins near random behavior and gradually becomes greedy. Evaluation
uses \(\epsilon=0\) for the learner. / 训练探索率从 1.0 逐步降到 0.05，评估关闭 learner 随机探索，避免把随机按键混入模型能力。

## 7. Opponent curriculum / 对手课程

Three modes are available:

- `random`: reproduce the easy baseline opponent.
- `snapshot`: play a frozen copy of the learner after warmup.
- `mixed`: choose random or snapshot once per episode.

为什么使用 frozen snapshot 而不是让同一个 online network 同时控制双方？如果两边参数每一步都更新，learner 面对的 opponent dynamics 变化太快。冻结一段时间让当前训练区间更接近平稳；定期刷新 snapshot 又能逐步提高对手强度。

Mixed training keeps some random opponents to preserve broad easy-state
coverage and reduce forgetting. / 混合 random 还能防止 agent 只适应某一代 snapshot。它不是完整 population-based self-play，但比 random-only baseline 更接近竞技评估。

## 8. Reward clipping / 奖励裁剪

Training can store \(\operatorname{sign}(r)\) while logs preserve raw episode
reward. Sign clipping controls target scale:

\[
\tilde r=\begin{cases}-1&r<0\\0&r=0\\1&r>0.\end{cases}
\]

好处是减少不同 reward magnitude 对 gradient 的影响；代价是丢失“一次得 5 分比 1 分更好”的幅度信息。它属于 Atari DQN 常见工程选择，应通过实验比较，而不是当成理论必需。

## 9. Evaluation protocol / 评估协议

Every evaluation uses held-out fixed seeds and greedy learner actions. Metrics
are recorded separately against random and snapshot opponents:

\[
\bar R=\frac1K\sum_{k=1}^K R_k,
\qquad
\sigma_R=sqrt{\frac1K\sum_k(R_k-\bar R)^2}.
\]

Final evaluation also reports min/max, win rate \(P(R>0)\), and draw rate
\(P(R=0)\). / 平均值衡量整体水平，标准差与最差结果衡量稳定性，胜率更贴近 tournament 目标。

The best checkpoint is selected by mean random performance before snapshots
exist, then by the average of random and snapshot mean returns. This makes the
selection rule explicit rather than choosing a visually attractive training
episode.

## 10. W&B evidence / W&B 实验证据

The optional W&B run stores configuration and histories for:

- raw training return and opponent type;
- epsilon and replay size;
- Huber loss, Q mean, target mean, absolute TD error, gradient norm;
- periodic random/snapshot evaluation;
- latest and best model artifact.

W&B 不改变算法。它建立“这个 checkpoint 是用什么参数、哪个 seed、哪种 opponent curriculum 训练出来的”证据链。正式运行可 online；网络或认证不稳定时使用 offline，之后 `wandb sync`。

## 11. Reproducibility and limits / 可重复性与限制

Python, NumPy, PyTorch, replay sampling, environment resets, and opponent
selection are seeded. CUDA kernels can still have nondeterministic behavior,
and one training seed is not enough for an algorithm-level claim.

Checkpoint 保存 online、target、opponent、optimizer、steps、episodes、best score 和配置。但 replay buffer 没有写入 checkpoint，以控制文件体积；resume 后会重新收集 warmup 数据。这一点必须在报告中明确。

The current opponent pool contains only one frozen snapshot. A stronger
tournament system would retain several historical snapshots and sample among
them, preventing cyclic forgetting and measuring exploitability across a
population.

## 12. Commands / 运行命令

```bash
cd projects/boxing-dqn
python -m unittest discover -s tests -v

python src/dqn_boxing.py \
  --episodes 200 \
  --device cuda \
  --opponent mixed \
  --wandb-mode offline

python src/evaluate_boxing.py \
  --checkpoint checkpoints/day4_boxing_best.pt \
  --eval-episodes 100 \
  --device cuda
```

Generated checkpoints, CSV files, W&B runs, notes, and reports remain ignored.
Only source, tests, requirements, and project README are uploaded. / GitHub 只上传可复现训练所需代码，不上传私人课堂资料、大模型文件和本地实验日志。

## 13. Source chain / 源码链

1. Instructor repository: environment/setup tutorial, no published Boxing trainer at the inspected commit.
2. Day 3 local baseline: compact random-opponent DQN, preserved unchanged under `materials/day-04/boxing-baseline/source/`.
3. Day 4 tracked implementation: Double-Dueling DQN plus opponent curriculum and evaluation.

老师仓库是环境来源，不把我们自写的 Day 3 baseline 冒充老师代码。详细逐段差异、公式和运行结果见被忽略的 `reports/day-04/source-vs-modified.md`。

## 14. Training results / 训练结果

The accepted v3 run completed 30 episodes and 75,000 environment steps on the
GTX 1650 Ti. The replay CSV passed the monotonicity check: episode identifiers
were exactly 1--30 and total steps never repeated or decreased. / 正式 v3 run
在 GTX 1650 Ti 上完成 30 局、75,000 个环境步；CSV 中 episode 恰好为
1--30，步数严格递增，没有混入旧进程的数据。

| Measurement / 指标 | Result / 结果 |
|---|---:|
| Training return, all 30 episodes / 全部训练局均值 | -0.20 |
| First five episode mean / 前五局均值 | -0.20 |
| Last five episode mean / 后五局均值 | -3.00 |
| Best/worst single training return / 最好与最差单局 | 12 / -8 |
| Best selection checkpoint / 最佳选模点 | episode 20, step 50,000 |
| Best 3-seed selection score / 最佳三种子选模分数 | -0.3333 |

The best checkpoint was then evaluated on unseen seeds 20000--20019, with 20
episodes per opponent and the same 2,500-step limit. / 最佳 checkpoint 再用未参与
选模的种子 20000--20019 评估；每类对手 20 局，并保持相同的 2,500 步上限。

| Opponent / 对手 | Mean | Std | Min | Max | Win | Draw |
|---|---:|---:|---:|---:|---:|---:|
| Random / 随机 | -0.05 | 1.8296 | -3 | 6 | 30% | 25% |
| Frozen snapshot / 冻结快照 | 0.00 | 0.0000 | 0 | 0 | 0% | 100% |

Interpretation / 结论：the complete CUDA, replay, checkpoint, opponent,
evaluation, and W&B pipeline works, but this short run did **not** demonstrate
stable convergence. The final five training episodes were worse than the first
five, and a zero return against the frozen snapshot means stalemate rather than
superiority. More training, multiple seeds, and opponent-pool ablations are
needed before making an algorithmic claim. / 工程链路已经跑通，但短预算没有证明
稳定收敛；后五局反而更差，对冻结快照全为 0 代表僵局，不代表战胜对手。需要更长
训练、多随机种子和 opponent-pool 消融实验后，才能判断方法是否真正有效。

## 15. Tournament-eve continuation and final policy / 赛前续训与最终策略

The episode-20 checkpoint was resumed for a longer run ending at episode 200
and 500,000 total environment steps. W&B ran in offline mode, while checkpoints
and CSV evidence stayed local. Periodic two-opponent selection scores were:

第 20 局 checkpoint 随后继续训练到第 200 局，总计 500,000 个环境步。W&B
使用 offline 模式，checkpoint 与 CSV 证据只保存在本地。每 20 局进行一次
random/snapshot 双对手评估：

| Episode | Step | Random mean | Snapshot mean | Selection score |
|---:|---:|---:|---:|---:|
| 40 | 100,000 | 0.6 | 0.0 | 0.3 |
| 60 | 150,000 | -2.0 | 0.0 | -1.0 |
| 80 | 200,000 | 1.2 | 0.0 | **0.6** |
| 100 | 250,000 | -1.2 | 0.0 | -0.6 |
| 120 | 300,000 | -0.4 | 0.0 | -0.2 |
| 140 | 350,000 | -2.2 | 0.0 | -1.1 |
| 160 | 400,000 | -0.6 | 0.0 | -0.3 |
| 180 | 450,000 | 0.8 | 0.0 | 0.4 |
| 200 | 500,000 | -5.0 | 0.0 | -2.5 |

Episode 80 was the best checkpoint *inside the continuation run*, but the
tournament package was selected by a stricter held-out comparison of episodes
20, 40, and 80. Each candidate played the same 20 unseen seeds from both player
positions. Define the role-balanced score as

续训内部的最佳点是第 80 局，但 tournament 模型没有直接采用它。我们把第 20、
40、80 局三个候选模型放在相同的 20 个未见种子上，并分别作为双方角色评估。
角色平衡分数定义为

\[
S_{role}=\frac{\bar R_{first}+\bar R_{second}}{2}.
\]

| Candidate | First-player mean | Second-player mean | Role-balanced mean |
|---|---:|---:|---:|
| Episode 20 | -0.50 | -0.50 | **-0.50** |
| Episode 40 | -0.80 | -0.75 | -0.775 |
| Episode 80 | -1.70 | -0.15 | -0.925 |

Episode 20 was retained because it had the best role-balanced held-out mean.
The result is intentionally modest: longer training improved one narrow
selection checkpoint but did not improve robust generalization. This is an
example of why more environment steps do not guarantee a better final policy.

最终保留第 20 局，因为它的跨角色 held-out 均值最好。这个结论不夸大效果：更长
训练曾提高某个小型选模分数，却没有提高稳健泛化；训练步数更多不等于最终策略必然
更强。两个角色收到的像素 observation 相同，部署 agent 也不能可靠地从画面判断自己
是 `first_0` 还是 `second_0`，因此选择跨角色更稳健的模型比偏向单一角色更合理。

The final tournament package keeps the instructor interface unchanged:
`sample_agent/agent_template.py`, `sample_agent/__init__.py`, and
`sample_agent/policy_weights.pt`. It performs CPU-only greedy inference, uses
the exact training preprocessing and four-frame stack, and resets its temporal
state when `get_action(None)` is called. / 最终提交包保持老师模板的类名、方法签名和
目录结构不变，只在 `Agent` 内加载选定权重并执行 CPU greedy inference。
