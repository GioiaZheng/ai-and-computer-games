# Course Projects

This directory collects the practical implementations and tournament agents
developed during **Artificial Intelligence for Computer Games**.

| Project | Environment | Main method | Included outcome |
| --- | --- | --- | --- |
| `boxing-dqn` | PettingZoo Atari Boxing | Double-Dueling DQN and self-play | Training and evaluation code |
| `boxing-agent` | PettingZoo Atari Boxing | Shared-role Double-Dueling DQN | Exact tournament agent and weights |
| `combat-tank-agent` | PettingZoo Atari Combat Tank | Behavioral cloning, exploration curriculum, and role-specialized PPO | Training code, report, and final agent |

The two final agents preserve the instructor-specified observation wrappers and
`Agent(env).get_action(state)` interface. They do not modify game rules,
rendering, rewards, or evaluation timing.

## Consolidated Projects

The Boxing and Combat Tank tournament agents were originally developed as
standalone projects. Their final agents, trained weights, tests, and supporting
documentation are now maintained here alongside the course notes, notebooks,
and practical work.

## Attribution

The course setup and submission contracts are based on
[Hautamaki-lab/Summer-School-2026](https://github.com/Hautamaki-lab/Summer-School-2026).
This repository contains the team's independent implementation and does not
redistribute instructor lecture PDFs, slides, or third-party agents.
