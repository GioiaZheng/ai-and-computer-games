# Atari Boxing Tournament Agent

Tournament submission for the University of Eastern Finland course
**Artificial Intelligence for Computer Games**. The agent placed **fourth** in
the 2026 course tournament.

> The project follows the environment and submission interface published in
> [Hautamaki-lab/Summer-School-2026](https://github.com/Hautamaki-lab/Summer-School-2026).
> This is an independent coursework implementation, not a fork of the course
> repository.

## Project at a Glance

| Item | Details |
| --- | --- |
| Environment | PettingZoo Atari `boxing_v2` |
| Learning method | Double-Dueling DQN |
| Observation | `uint8 (84, 84, 6)` |
| Action space | 18 discrete Atari actions |
| Player handling | One shared policy for both roles |
| Inference | Greedy Q-values with a short anti-stall fallback |
| Tournament result | 4th place |

The convolutional policy contains 1,697,971 trainable parameters. It loads the
included checkpoint on CUDA when available and otherwise uses CPU.

## Repository Layout

```text
boxing-agent/
|-- sample_agent/
|   |-- __init__.py             Required package entry point
|   |-- agent_template.py       Network and tournament Agent interface
|   `-- policy_weights.pt       Submitted trained checkpoint
|-- docs/
|   `-- method.md               Architecture and training notes
|-- tests/
|   `-- test_agent.py           CPU inference and package smoke tests
|-- requirements.txt
`-- README.md
```

Only the evaluation-time agent is included. Training checkpoints, W&B runs,
lecture material, and local tournament tools are intentionally excluded.

## Evaluation Contract

The agent was trained and evaluated with the instructor-specified wrapper
order:

```python
from pettingzoo.atari import boxing_v2
import supersuit as ss

env = boxing_v2.parallel_env(render_mode="human")
env = ss.max_observation_v0(env, 2)
env = ss.frame_skip_v0(env, 4)
env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)
env = ss.agent_indicator_v0(env, type_only=False)
```

The evaluator constructs `Agent(env)` and calls `get_action(state)`. The
submission does not alter the environment, wrapper order, rendering, reward,
or timing.

```python
from sample_agent.agent_template import Agent

agent = Agent(env)
action = agent.get_action(state)
```

Calling `get_action(None)` resets episode-local controller state.

## Local Verification

Create a clean Python 3.12 environment, install the two inference dependencies,
and run the smoke tests:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests load the packaged weights, verify the network architecture, execute
a CPU forward pass, check the 18-action output contract, and validate reset and
input-shape behavior. They do not require Atari ROMs.

## Method Summary

- Double DQN targets reduce action-value overestimation.
- Dueling value and advantage heads separate screen value from action choice.
- Random, frozen snapshot, self-play, and external opponents broaden training.
- Both boxer roles contribute experience to one shared convolutional policy.
- Fixed-seed two-role evaluations are used for checkpoint comparison.
- A bounded anti-stall fallback interrupts indefinite action repetition while
  leaving the official environment unchanged.

See [docs/method.md](docs/method.md) for the model equation, architecture, and
known limitations.

## Scope and Attribution

Only the team's implementation and trained submission are distributed here.
Lecture slides, PDFs, instructor source files, and other course materials are
not redistributed. PettingZoo, SuperSuit, ALE, and Atari content remain subject
to their own licenses and terms.

## Team

- Gioia Zheng
- ZeYang Fu
- Gankhulug Bayaraa
