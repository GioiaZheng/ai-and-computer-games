# Course Overview and Source Status / 课程概览与资料状态

This note preserves the course overview, source status, logistics, and project context for the University of Eastern Finland course "Artificial Intelligence (AI) for Computer Games".

本页保存芬兰东部大学课程的日程、资料版本与项目边界。2026 年项目以 Atari Boxing 为准；旧 PDF 中出现的往年游戏只用于讲解算法，不能覆盖今年项目要求。

Related local notes:

- [lecture_notes.md](lecture_notes.md): integrated course notes
- [notebook_index.md](notebook_index.md): notebook inventory and conceptual map
- [day-03/imitation-learning.md](day-03/imitation-learning.md): BC, IRL, GAIL, and AIRL
- [day-03/mpe2-practical.md](day-03/mpe2-practical.md): Simple Spread, IQL, and centralized Q-learning

## Source Status

This version was created from the supplied course page, the current 2026 introductory lecture PDF, older lecture PDFs that the user confirmed are reused this year, the automatic differentiation survey paper, manual lecture context, and `gradient_descent_regression.ipynb`.

Important source discipline:

资料使用原则：当前 2026 课件优先于旧课件；课堂照片用于补充当天实际讲解；学习补充必须与老师明确讲过的内容区分；收到的脚本和权重原样保存在 `materials/`，我们自己的修正另行说明。

- The current course project is Boxing Game / Atari Boxing, not last year's Minimal Fighting Game.
- `2.pdf` duplicates the RL algorithms lecture, and `3.pdf` duplicates the Introduction to Reinforcement Learning lecture.
- Day 1 material covers intro, ML basics, optimization, automatic differentiation, single-agent RL foundations, and RL algorithms.
- Day 2 material is multi-agent reinforcement learning. The Backstabbr/Diplomacy screenshot is kept as a Day 2 example source.
- Wednesday and project sections were listed as not yet available on the supplied course page at the time these notes were created.

## Course Logistics

## 课程安排中文摘要

- 负责人：Ville Hautamäki；授课教师还包括 Anssi Kanervisto、Ekaterina Amozova、Nima Hadavi、Mikko Turunen、Janne Laakkonen。
- 地点：Joensuu，Metria building，Auditorium M101。
- 形式：Pass/Fail；学生自带电脑；项目最多三人组队。
- Day 1：ML、优化、单智能体 RL 与研究报告。
- Day 2：MARL 基础、Minecraft 与简单环境实践。
- Day 3：MARL 续讲、模仿学习与 MPE2 实践。
- Day 4：Boxing 项目训练。
- Day 5：最终 tournament 与结果总结。

- Lecturers: Ville Hautamaki, Anssi Kanervisto, Ekaterina Amozova, Nima Hadavi, Mikko Turunen, and Janne Laakkonen.
- Teacher in charge: Ville Hautamaki.
- Dates: Monday, 10 August to Friday, 14 August.
- Location: Auditorium M101, Metria building, Joensuu.
- Assessment: Pass/Fail.
- Practical format: students bring their own devices, work in groups of up to 3, and train a game-playing software agent for a final tournament.
- Project: Boxing Game / Atari Boxing.
- Course book: Stefano V. Albrecht, Filippos Christianos, and Lukas Schafer, _Multi-Agent Reinforcement Learning: Foundations and Modern Approaches_, MIT Press, 2024.
