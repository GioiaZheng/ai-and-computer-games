# Training Curriculum

## Why Separate Navigation and Combat?

Combat Tank has sparse official rewards. A policy can execute thousands of
actions without hitting either tank, so a single end-to-end objective often
collapses to a small set of repeated actions. The curriculum isolates skills
while preserving the official maze and action timing.

## Stage 1: Initialization

Behavioral cloning uses role-specific demonstrations to initialize useful
movement and firing actions. It is an initialization only: a fixed route is not
considered a complete policy.

## Stage 2: Frontier Exploration

The map is discretized into coarse cells. During one episode, only the first
entry into a cell receives a novelty reward. Counts persist across episodes in
each worker so repeatedly visited routes become less valuable:

\[
r_{visit}(c)=\frac{\beta}{\sqrt{N(c)}}.
\]

Pure movement has a very small reward. Long periods without a new cell, no
physical displacement, and dominant repeated actions are penalized. Both roles
are trained and evaluated independently.

## Stage 3: Tactical Fine-Tuning

An official `-1` identifies an exposed position. Training temporarily records
that location and rewards real displacement away from it while penalizing an
immediate return. An official `+1` opens only a short confirmation window; the
policy must then use new observations rather than repeatedly firing at the old
position.

These event signals are training aids. They do not change official evaluation.

## Stage 4: Opponent Mixture

Episodes sample random, scripted, and frozen learned opponents. Fixed opening
prefixes are disabled for the final stages so the policy sees the same initial
state distribution as tournament evaluation.

## Selection Criteria

A checkpoint is not selected by shaped return alone. Required measurements
include:

- official score, hits, and received hits;
- performance from `first_0` and `second_0`;
- maze-cell coverage and idle fraction;
- deterministic action coverage;
- results across multiple seeds and opponent types.
