"""Build the PDF report for the central Q-learning final work."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
REPORT_PATH = REPORT_DIR / "central_q_learning_report.pdf"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(rows: list[dict[str, str]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def make_table(data: list[list[str]], widths: list[float] | None = None) -> Table:
    table = Table(data, colWidths=widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#142033")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c0cc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_report() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    eval_rows = read_csv(RESULTS_DIR / "evaluation_summary.csv")

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyTight",
            parent=styles["BodyText"],
            fontSize=10.5,
            leading=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
        )
    )

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Central Q-Learning for a Cooperative MARL Game",
    )

    story = []
    story.append(Paragraph("Central Q-Learning for a Cooperative MARL Game", styles["Title"]))
    story.append(Paragraph("AI for Computer Games - Final / Extra Work", styles["Heading2"]))
    story.append(
        Paragraph(
            "This report describes a cooperative multi-agent reinforcement learning experiment "
            "trained with central Q-learning. The central learner acts as a super brain: it "
            "observes the full joint state and chooses a joint action for both agents.",
            styles["BodyTight"],
        )
    )

    story.append(Paragraph("1. Task and Environment", styles["Heading1"]))
    story.append(
        Paragraph(
            "The selected cooperative game is a self-contained two-agent gridworld. The game is "
            "small enough for tabular Q-learning, but still multi-agent because the state and "
            "action are joint objects. Agent A and Agent B move on a 5x5 grid. Agent A must "
            "reach target A at (0, 4), Agent B must reach target B at (4, 0), and the episode "
            "succeeds only when both agents are on their assigned targets at the same time.",
            styles["BodyTight"],
        )
    )
    story.append(
        make_table(
            [
                ["Component", "Definition"],
                ["State", "(row_A, col_A, row_B, col_B)"],
                ["Individual actions", "stay, up, down, left, right"],
                ["Joint action", "(action_A, action_B), giving 5 x 5 = 25 choices"],
                ["Terminal success", "Agent A on target A and Agent B on target B simultaneously"],
                ["Episode limit", "40 environment steps"],
            ],
            widths=[4.2 * cm, 11.2 * cm],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        Paragraph(
            "The reward is shared. Each step has a small penalty, moving closer to the targets "
            "adds shaped reward, collisions are penalized, and simultaneous success gives a "
            "large positive terminal reward.",
            styles["BodyTight"],
        )
    )
    story.append(
        make_table(
            [
                ["Reward term", "Value"],
                ["Step penalty", "-0.02"],
                ["Distance shaping", "+0.05 x reduction in total Manhattan distance"],
                ["Collision penalty", "-0.10"],
                ["Terminal success", "+10.0"],
            ],
            widths=[5.2 * cm, 10.2 * cm],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("2. Central Q-Learning Method", styles["Heading1"]))
    story.append(
        Paragraph(
            "Central Q-learning treats the two-agent system as one centralized learner over "
            "joint states and joint actions. This is the super-brain formulation requested in "
            "the assignment.",
            styles["BodyTight"],
        )
    )
    story.append(
        Paragraph(
            "The tabular update is: Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]. "
            "Here s is the joint state, a is the joint action, r is the shared reward, alpha is "
            "the learning rate, and gamma is the discount factor.",
            styles["BodyTight"],
        )
    )
    story.append(
        make_table(
            [
                ["Hyperparameter", "Value"],
                ["Training episodes", "8000 per seed"],
                ["Random seeds", "5"],
                ["Evaluation episodes", "300 per seed"],
                ["Learning rate alpha", "0.20"],
                ["Discount factor gamma", "0.95"],
                ["Exploration", "epsilon-greedy, decayed from 1.0 toward 0.05"],
            ],
            widths=[5.2 * cm, 10.2 * cm],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("3. Results", styles["Heading1"]))
    result_table = [["Seed", "Success rate", "Avg. return", "Avg. steps", "Avg. collisions", "Visited Q states"]]
    for row in eval_rows:
        result_table.append(
            [
                str(int(float(row["seed"]))),
                f"{float(row['eval_success_rate']):.3f}",
                f"{float(row['eval_avg_return']):.3f}",
                f"{float(row['eval_avg_steps']):.2f}",
                f"{float(row['eval_avg_collisions']):.2f}",
                f"{float(row['q_states_visited']):.0f}",
            ]
        )
    result_table.append(
        [
            "Mean",
            f"{mean(eval_rows, 'eval_success_rate'):.3f}",
            f"{mean(eval_rows, 'eval_avg_return'):.3f}",
            f"{mean(eval_rows, 'eval_avg_steps'):.2f}",
            f"{mean(eval_rows, 'eval_avg_collisions'):.2f}",
            f"{mean(eval_rows, 'q_states_visited'):.0f}",
        ]
    )
    story.append(make_table(result_table))
    story.append(Spacer(1, 0.35 * cm))
    story.append(
        Paragraph(
            "Across five random seeds, the final greedy policy solved all evaluation episodes. "
            "The learned policy also avoided collisions in evaluation. The average episode "
            "length was about five steps, which is close to the shortest practical paths for "
            "many sampled starts.",
            styles["BodyTight"],
        )
    )
    curve_path = RESULTS_DIR / "training_curve.png"
    story.append(Image(str(curve_path), width=16.0 * cm, height=11.33 * cm))
    story.append(
        Paragraph(
            "Figure: 100-episode moving average over five seeds. The upper panel shows return; "
            "the lower panel shows training success rate.",
            styles["SmallNote"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("4. Discussion", styles["Heading1"]))
    story.append(
        Paragraph(
            "The experiment shows that central Q-learning is effective when the cooperative "
            "game has a small discrete joint state-action space. The central learner sees the "
            "full state and therefore does not face partial observability or decentralized "
            "coordination problems. In this setting, a tabular Q function can store useful "
            "values for the visited joint states and quickly learn coordinated paths.",
            styles["BodyTight"],
        )
    )
    story.append(
        Paragraph(
            "The main limitation is scalability. With n agents, the joint action space grows "
            "multiplicatively. Here two agents with five actions each give 25 joint actions. "
            "Three agents would already give 125 joint actions, and larger gridworlds would "
            "also increase the number of joint states. This is why practical MARL often uses "
            "function approximation, value decomposition, actor-critic methods, or centralized "
            "training with decentralized execution.",
            styles["BodyTight"],
        )
    )
    story.append(
        Paragraph(
            "Another limitation is reward shaping. The terminal reward defines the real task, "
            "while the distance shaping makes learning faster. If the shaping term were poorly "
            "designed, the agent could learn behavior that improves the proxy but does not "
            "solve the intended cooperative task. This connects directly to the course theme "
            "that reward design and evaluation must be checked carefully.",
            styles["BodyTight"],
        )
    )

    story.append(Paragraph("5. Reproducibility", styles["Heading1"]))
    story.append(
        Paragraph(
            "Run the experiment with: python central_q_learning.py --episodes 8000 --seeds 5 "
            "--output-dir ../results. The script writes training_log.csv, evaluation_summary.csv, "
            "and training_curve.png. The report was generated from those outputs.",
            styles["BodyTight"],
        )
    )
    story.append(
        Paragraph(
            "The environment is implemented directly in the submitted code rather than loaded "
            "from PettingZoo. This keeps the final work self-contained while still satisfying "
            "the central cooperative MARL requirement.",
            styles["SmallNote"],
        )
    )

    doc.build(story)
    print(REPORT_PATH)


if __name__ == "__main__":
    build_report()
