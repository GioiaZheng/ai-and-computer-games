# Notebook Index

# Notebook 索引说明

本页记录 notebook 的主题、公式、输出图和对应讲义位置。Notebook 用于可重复实验，主讲义用于解释概念；两者不应简单复制整段内容。

## Incorporated Notebooks

| Notebook | Main topic | Key equations or algorithms | Outputs to interpret | Linked note sections |
| --- | --- | --- | --- | --- |
| `notebooks/gradient_descent_regression.ipynb` | Linear regression, logistic regression, batch gradient descent | MSE, binary cross entropy, sigmoid, analytic gradients, parameter updates | Fitted-line evolution, loss vs iteration, loss contours, decision-boundary evolution | Sections 3, 4, 5 |

## Notebook Summary

## Notebook 中文摘要

`gradient_descent_regression.ipynb` 共 23 个 cell，先用线性回归展示拟合直线、loss 曲线和参数在 contour 上的下降路径，再用逻辑回归展示 decision boundary 与 cross-entropy。重点不是只运行出图，而是把每张图与 objective、gradient 和 parameter update 对应起来。

`gradient_descent_regression.ipynb` contains 23 cells: 10 markdown cells, 13 code cells, and stored outputs. It is organized into:

- Part 1: Linear Regression
- Visualization 1: fitted line evolving over training
- Visualization 2: loss/cost vs iteration
- Visualization 3: loss contour with gradient descent path
- Part 2: Logistic Regression
- Visualization 1: decision boundary evolving over training
- Visualization 2: cross-entropy loss vs iteration
- Visualization 3: logistic loss contour with gradient descent path
- Ideas for class discussion / exercises

## Integration Rules

## 整合原则

- 保持 notebook 可运行，讲义则解释“为什么这样写”。
- 图表必须说明横纵轴、数据来源与能够支持的结论。
- 不把训练集 loss 当成泛化能力；需要单独 evaluation。
- 公式中的 shape、batch 维、dtype 和代码 API 应明确对应。

- Do not copy entire notebook cells into the lecture notes.
- Explain the purpose of each experiment.
- Link code to the corresponding mathematical objective, loss, update rule, or RL concept.
- Mention outputs and plots only when they help interpret the experiment.
- Keep notebook code runnable and keep the notes readable.
