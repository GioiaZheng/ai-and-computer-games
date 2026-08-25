# Method Notes

## Problem

Atari Boxing is a two-player, zero-sum control problem with sparse and delayed
effects. Movement changes distance and alignment, while successful punches
depend on timing and relative position. A policy must therefore learn both
short combat sequences and recovery behavior after either boxer moves away.

The tournament evaluator supplies the same processed observation to every
submission. No game-state coordinates or privileged emulator variables are
used by this agent.

## Observation Pipeline

The official observation has shape `84 x 84 x 6`:

- four grayscale image frames retain short-term motion information;
- two agent-indicator planes identify the controlled player role;
- each `uint8` channel is converted to channel-first format and divided by 255.

The role indicators allow one network to represent different behavior for the
left and right boxer without changing the public `Agent` interface.

## Network Architecture

| Layer | Configuration |
| --- | --- |
| Convolution 1 | 6 to 32 channels, 8 x 8 kernel, stride 4, ReLU |
| Convolution 2 | 32 to 64 channels, 4 x 4 kernel, stride 2, ReLU |
| Convolution 3 | 64 to 64 channels, 3 x 3 kernel, stride 1, ReLU |
| Shared layer | 3,136 to 512, ReLU |
| Value head | 512 to 1 |
| Advantage head | 512 to 18 |

The two heads are combined as

\[
Q(s,a)=V(s)+A(s,a)-\frac{1}{|\mathcal A|}\sum_{a'}A(s,a').
\]

The value head estimates the quality of the screen as a whole. The advantage
head measures how much better or worse each of the 18 actions is in that state.
The complete network has 1,697,971 trainable parameters.

## Double DQN Update

The online network selects the next action and a delayed target network
evaluates that action:

\[
y_t=r_t+\gamma(1-d_t)Q_{\theta^-}\left(s_{t+1},
\arg\max_a Q_\theta(s_{t+1},a)\right).
\]

Here, `d_t` marks a terminal transition. Separating selection from evaluation
reduces the positive bias that appears when one noisy estimate performs both
operations. Training used replay memory, Huber loss, gradient clipping, an
epsilon-greedy behavior policy, and periodic target-network updates.

## Opponent Curriculum and Selection

Training opponents included random policies, frozen checkpoints, self-play, and
selected external agents. This matters because a policy trained against one
stationary opponent can obtain a high training score while learning a brittle
opening sequence.

Candidate checkpoints were inspected from both player roles. Official reward,
action use, repeated-action counts, full-match playback, and results against
more than one opponent were considered together. Training loss alone was not a
selection criterion.

## Evaluation-Time Anti-Stall Controller

A deterministic DQN can repeat one locally preferred action indefinitely after
entering an unfamiliar state. The submission counts consecutive greedy
actions. After 60 repetitions, it spends at most eight decisions selecting
high-ranked actions that were not used recently, then returns control to the
network.

The controller changes only the final discrete action. It does not modify the
observation, reward, frame skip, render mode, game rules, or emulator state.

## Reproducibility Boundary

This public repository is the compact tournament artifact rather than a full
training archive. It contains the exact inference model and selected weights,
plus tests for the package contract. Large replay buffers, intermediate
checkpoints, W&B logs, course files, and third-party agents are excluded.

## Known Limitations

- Four stacked frames provide short motion context but no recurrent memory.
- The anti-stall fallback mitigates repeated actions; it does not learn a new
  recovery policy.
- The final checkpoint reflects the available opponent pool and can still be
  exploited by behavior not represented during training.
- Tournament placement is a small-sample outcome, not a general performance
  guarantee.
