# Final Work: Central Q-Learning for a Cooperative MARL Game

This folder contains the extra/final work submission for the AI for Computer Games course.

The task is to train a central Q-learning "super brain" agent for a cooperative MARL game and report the results.

## Contents

```text
final-work/
|-- code/
|   |-- central_q_learning.py
|   `-- requirements.txt
|-- results/
|   |-- training_curve.png
|   |-- training_log.csv
|   `-- evaluation_summary.csv
|-- report/
|   `-- central_q_learning_report.pdf
`-- README.md
```

## Run

```bash
cd final-work/code
python central_q_learning.py --episodes 8000 --seeds 5 --output-dir ../results
```

The script trains a tabular central Q-learning agent on a two-agent cooperative gridworld, evaluates greedy policies, and writes plots/tables to `results/`.

## Submission

Submit the generated zip file containing this folder, including the code and PDF report.
