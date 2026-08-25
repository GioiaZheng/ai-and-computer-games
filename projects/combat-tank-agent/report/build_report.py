"""Build the Ali0x01 Combat Tank final report."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
ASSET_DIR = REPORT_DIR / "assets"
OUTPUT = REPORT_DIR / "Ali0x01_combat_tank_report.pdf"

BLUE_FILL = colors.HexColor("#E6EDF7")
BLUE_LINE = colors.HexColor("#A9B9CE")
TEXT = colors.HexColor("#111111")
MUTED = colors.HexColor("#4B5563")


def make_score_chart() -> Path:
    """Plot held-out scores used for checkpoint selection."""
    labels = [
        "First base",
        "First exploration",
        "First overnight",
        "Second base",
        "Second specialist",
        "Second exploit",
    ]
    random_scores = [0.20, 0.40, -0.40, -0.40, -0.80, -0.20]
    frozen_scores = [0.20, 0.00, 0.20, 0.40, 1.20, 0.60]
    width, height = 1428, 595
    chart = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(chart)
    font = ImageFont.load_default()
    plot_left, plot_top, plot_right, plot_bottom = 110, 55, 1395, 455
    min_value, max_value = -1.2, 1.6

    def y_position(value: float) -> float:
        fraction = (max_value - value) / (max_value - min_value)
        return plot_top + fraction * (plot_bottom - plot_top)

    for value in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
        y = y_position(value)
        draw.line((plot_left, y, plot_right, y), fill="#D7DCE3", width=1)
        draw.text((55, y - 7), f"{value:+.1f}", fill="#30343B", font=font)
    zero_y = y_position(0.0)
    draw.line((plot_left, zero_y, plot_right, zero_y), fill="#60646C", width=2)

    group_width = (plot_right - plot_left) / len(labels)
    bar_width = 44
    for index, label in enumerate(labels):
        center = plot_left + group_width * (index + 0.5)
        for offset, value, fill in (
            (-bar_width, random_scores[index], "#3F7CAC"),
            (0, frozen_scores[index], "#D9822B"),
        ):
            x0 = center + offset
            x1 = x0 + bar_width
            y = y_position(value)
            draw.rectangle((x0, min(y, zero_y), x1, max(y, zero_y)), fill=fill)
            draw.text((x0 + 5, y - 17 if value >= 0 else y + 4), f"{value:+.1f}", fill="#20242A", font=font)
        words = label.split()
        draw.text((center - 58, 475), " ".join(words[:-1]), fill="#20242A", font=font)
        draw.text((center - 58, 493), words[-1], fill="#20242A", font=font)

    draw.rectangle((110, 15, 130, 35), fill="#3F7CAC")
    draw.text((138, 19), "Random opponent", fill="#20242A", font=font)
    draw.rectangle((300, 15, 320, 35), fill="#D9822B")
    draw.text((328, 19), "Frozen PPO opponent", fill="#20242A", font=font)
    draw.text((12, 205), "Mean official score", fill="#20242A", font=font)
    output = ASSET_DIR / "heldout_scores.png"
    chart.save(output)
    return output


def make_match_strip() -> Path:
    """Combine three official-game frames into one compact visual."""
    frame_paths = [
        ROOT / "results" / "official-frames" / "frame-0000.png",
        ROOT / "results" / "official-frames" / "frame-0500.png",
        ROOT / "results" / "official-frames" / "frame-1000.png",
    ]
    labels = ["Opening", "Mid-game", "Later state"]
    frames = [Image.open(path).convert("RGB").resize((240, 384)) for path in frame_paths]
    canvas = Image.new("RGB", (780, 430), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (frame, label) in enumerate(zip(frames, labels)):
        x = 10 + index * 260
        canvas.paste(frame, (x, 28))
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (240 - (bbox[2] - bbox[0])) / 2, 8), label, fill="black", font=font)
    output = ASSET_DIR / "official_match_strip.png"
    canvas.save(output)
    return output


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            textColor=TEXT,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=TEXT,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=TEXT,
            spaceBefore=9,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=TEXT,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=TEXT,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            textColor=MUTED,
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "TableText",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=9.8,
            textColor=TEXT,
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=9.8,
            textColor=colors.HexColor("#243247"),
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10,
            textColor=MUTED,
            spaceBefore=3,
            spaceAfter=7,
        ),
    }


def p(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def table(data, widths, s, repeat_rows=1):
    formatted = []
    for row_index, row in enumerate(data):
        style = s["table_head"] if row_index == 0 else s["table"]
        formatted.append([p(str(cell), style) for cell in row])
    result = Table(formatted, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE_FILL),
                ("GRID", (0, 0), (-1, -1), 0.45, BLUE_LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def build():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    score_chart = make_score_chart()
    match_strip = make_match_strip()
    s = styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Ali0x01 - Combat Tank Agent",
        author="Gioia Zheng, ZeYang Fu, Gankhulug Bayaraa",
    )
    story = []

    story += [
        p("Ali0x01 - Combat Tank Agent", s["title"]),
        p("AI for Computer Games - Final Project Report", s["subtitle"]),
        table(
            [
                ["Group member", "Contact"],
                ["Gioia Zheng", "zheng.2040439@studenti.uniroma1.it"],
                ["ZeYang Fu", "zeyangfu26@gmail.com"],
                ["Gankhulug Bayaraa", "gbayaraa@uef.fi"],
            ],
            [62 * mm, 108 * mm],
            s,
        ),
        Spacer(1, 5),
        p("1. Introduction", s["h1"]),
        p(
            "The task was to train an agent for the two-player Atari Combat Tank game and submit it through the instructor-provided Agent interface. The tournament configuration contains a maze, visible tanks, no invisibility, and no billiard-style bullet rebounds. Each player receives +1 for hitting the opponent and -1 when hit. The main difficulty is that useful combat events are sparse: a policy can execute thousands of actions without receiving a non-zero reward.",
            s["body"],
        ),
        p(
            "Our objective was therefore broader than learning to shoot. The agent had to leave its starting corridor, navigate around fixed walls, find or reacquire a moving opponent, and continue acting after either tank was displaced by a hit. It also had to work from both player roles under the exact evaluation wrappers.",
            s["body"],
        ),
        p("Environment used for all final evaluations", s["h2"]),
        table(
            [
                ["Component", "Final setting"],
                ["Game", "PettingZoo Atari combat_tank_v2, parallel API"],
                ["Rules", "has_maze=True, is_invisible=False, billiard_hit=False"],
                ["Observation", "84 x 84 grayscale; max over 2 frames; 4-frame stack; 2 role indicators"],
                ["Action space", "Discrete(18)"],
                ["Time handling", "Frame skip 4; reward clipped to [-1, 1]"],
                ["Official reward", "+1 hit, -1 received hit, otherwise 0"],
            ],
            [48 * mm, 122 * mm],
            s,
        ),
        p(
            "The wrapper order and game settings above were kept identical in training, evaluation, visual playback, and packaging. Training-only reward shaping never changed tournament scoring.",
            s["small"],
        ),
    ]

    story += [
        PageBreak(),
        p("2. Solution", s["h1"]),
        p("2.1 Initial idea", s["h2"]),
        p(
            "The initial plan was imitation learning followed by reinforcement learning. A scripted teacher demonstrated movement, turning, and firing, after which PPO would improve the policy from game rewards. This gave a useful debugging baseline, but a fixed teacher route did not cover the complete maze and transferred its route bias to the learner.",
            s["body"],
        ),
        p("2.2 PPO policy", s["h2"]),
        p(
            "The final learner used a convolutional PPO policy. Three convolution layers (32, 64, and 64 channels) encoded the six-channel official observation. A 512-unit projection produced logits for all 18 actions. PPO updates used the clipped objective L = E[min(r_t A_t, clip(r_t, 1-epsilon, 1+epsilon) A_t)], with epsilon=0.1 in the final branches. Training used CUDA and multiple parallel environments.",
            s["body"],
        ),
        p(
            "Because the two spawn positions and useful openings are asymmetric, one shared checkpoint was not consistently best for both roles. We therefore exported two role-specialized policies into one Agent. The two indicator planes appended by SuperSuit select the correct action head at inference time.",
            s["body"],
        ),
        p("2.3 Navigation shaping", s["h2"]),
        p(
            "The maze was divided into coarse position cells. During exploration stages, entering a new cell added beta/sqrt(N(c)) to the official reward, where N(c) is the lifetime visit count for that role and cell. Long periods without physical displacement received a small penalty. The complete training reward was r_train = r_official + r_visit - r_idle. These terms encouraged exploration but were set to zero during model selection.",
            s["body"],
        ),
        p("2.4 Opponent curriculum and inference", s["h2"]),
        p(
            "Training alternated between random opponents and frozen PPO snapshots. Later branches used mixed opponents and separate first-role and second-role fine-tuning. The learned PPO policies remained deliberately stochastic. Pure argmax inference collapsed some checkpoints to one repeated action, so the submitted Agent samples from the learned categorical distribution. This preserves the behavior used during training and uses all 18 legal actions.",
            s["body"],
        ),
        table(
            [
                ["Parameter", "Final branch value"],
                ["Optimizer / learning rate", "Adam / 1e-5 for long fine-tuning"],
                ["Rollout", "4 or 8 environments; 512 or 1024 steps per environment"],
                ["PPO epochs / batch", "4 epochs / 256 samples"],
                ["Discount / GAE", "gamma=0.99 / lambda=0.95"],
                ["Entropy coefficient", "0.005-0.02 depending on branch"],
                ["Overnight branches", "1.5M exploit + 1.5M diverse + 0.75M per role"],
            ],
            [58 * mm, 112 * mm],
            s,
        ),
    ]

    story += [
        PageBreak(),
        p("3. Development and Modifications", s["h1"]),
        p(
            "The project was developed by repeatedly watching full matches, measuring coverage and action use, and then changing one part of the curriculum. The table records the main observations and decisions.",
            s["body"],
        ),
        table(
            [
                ["Stage", "Observed problem", "Modification", "Outcome"],
                ["Random / basic PPO", "Sparse reward and no reliable contact", "Added scripted demonstrations and PPO initialization", "Agent moved and fired, but copied a narrow opening route"],
                ["Training vs random", "Random opponent often stayed near spawn; learner overfit to one side and route", "Added frozen-model and mixed opponents", "More realistic encounters, but some stationary policies remained"],
                ["Self-play", "Both tanks sometimes rotated and fired without changing position", "Added map-cell coverage and idle diagnostics", "Made the failure measurable instead of relying only on score"],
                ["Maze exploration", "17/26 cells but about 84% idle; only 3 actions dominated", "Tried frontier novelty and longer exploration", "Coverage improved in some runs, combat did not; not selected"],
                ["Frontier run", "Final policy collapsed to 2 actions and 6/8 cells", "Rejected final checkpoint and retained earlier candidates", "Confirmed that later checkpoints are not automatically better"],
                ["Waypoint teacher", "Top routes reached 4 waypoints, bottom routes only 2; high idle fraction", "Kept waypoint study as auxiliary pretraining only", "Avoided submitting route-specific behavior"],
                ["Long self-play", "4.5M extra steps did not consistently improve held-out score", "Selected checkpoints by opponent and role instead of training return", "Older role specialists beat several later checkpoints"],
                ["Tournament export", "Argmax produced repeated actions", "Sampled the PPO categorical distribution and exported two role heads", "18/18 stochastic action coverage and stronger peer matches"],
            ],
            [25 * mm, 48 * mm, 49 * mm, 48 * mm],
            s,
        ),
        Spacer(1, 6),
        p(
            "Two design rules followed from these experiments. First, shaped return was used for learning but never for promotion. Second, every candidate was checked from both roles and against more than one opponent. This prevented us from selecting a model that looked active but only exploited one stationary opponent.",
            s["body"],
        ),
    ]

    story += [
        PageBreak(),
        p("4. Numerical and Visual Results", s["h1"]),
        p("4.1 Held-out official evaluation", s["h2"]),
        p(
            "The table below reports five games per setting using the official unshaped reward. The frozen opponent is the 250k self-play checkpoint. Because five games are a small sample and scores have high variance, these values were used as screening evidence rather than precise estimates.",
            s["body"],
        ),
        table(
            [
                ["Role / candidate", "Random score", "Frozen score", "Cells vs random", "Cells vs frozen"],
                ["first_0 base", "+0.20 +/- 0.40", "+0.20 +/- 0.40", "15.6", "17.8"],
                ["first_0 exploration (selected)", "+0.40 +/- 1.02", "+0.00 +/- 0.63", "19.0", "19.4"],
                ["first_0 overnight diverse", "-0.40 +/- 0.80", "+0.20 +/- 0.40", "19.0", "18.0"],
                ["first_0 role specialist", "+0.20 +/- 0.98", "+0.00 +/- 0.63", "13.0", "13.2"],
                ["second_0 base", "-0.40 +/- 1.36", "+0.40 +/- 0.80", "19.6", "15.8"],
                ["second_0 specialist (selected)", "-0.80 +/- 1.17", "+1.20 +/- 1.47", "14.6", "19.0"],
                ["second_0 overnight exploit", "-0.20 +/- 0.75", "+0.60 +/- 1.36", "11.8", "14.2"],
            ],
            [50 * mm, 31 * mm, 31 * mm, 29 * mm, 29 * mm],
            s,
        ),
        Spacer(1, 5),
        RLImage(str(score_chart), width=168 * mm, height=70 * mm),
        p(
            "Figure 1. Mean official score for representative candidate branches. Blue bars use a random opponent; orange bars use the frozen PPO opponent. The selected role specialists were chosen for complementary behavior, not for one global average.",
            s["caption"],
        ),
    ]

    story += [
        PageBreak(),
        p("4.2 External peer matches", s["h2"]),
        p(
            "The selected first-role policy won all 3 peer-test games (+1 in each game). The selected second-role policy won 2 games and drew 1. These tests used the exact official observation pipeline and the peer's submitted Agent interface. They provided stronger opponent diversity than random-only testing, but the sample remains small.",
            s["body"],
        ),
        table(
            [
                ["Selected policy", "Games", "Wins", "Draws", "Losses"],
                ["first_0 exploration checkpoint", "3", "3", "0", "0"],
                ["second_0 role checkpoint", "3", "2", "1", "0"],
            ],
            [72 * mm, 24 * mm, 24 * mm, 24 * mm, 24 * mm],
            s,
        ),
        p("4.3 Visual behavior", s["h2"]),
        RLImage(str(match_strip), width=166 * mm, height=91.5 * mm),
        p(
            "Figure 2. Frames from an official maze match. The agent must rotate inside the narrow spawn corridor, pass the vertical wall opening, and reacquire the opponent after movement or a hit displacement.",
            s["caption"],
        ),
        p("4.4 Remaining limitations", s["h2"]),
        p(
            "The score is still low in many random-opponent games, and several episodes end in draws. Navigation coverage does not guarantee useful line of sight, while combat fine-tuning can erase exploration. The feed-forward four-frame policy also has no explicit long-term map memory. Finally, stochastic sampling is necessary for the submitted policy; deterministic argmax diagnostics reveal severe action collapse in some checkpoints. These limitations are reported because they materially affect tournament reliability.",
            s["body"],
        ),
    ]

    story += [
        PageBreak(),
        p("5. Conclusions", s["h1"]),
        p(
            "We reproduced the instructor's Combat Tank pipeline and trained a role-specialized PPO agent without changing the game rules or evaluation interface. The final solution combines a CNN policy, mixed-opponent training, exploration-only shaping, role-specific checkpoint selection, and stochastic tournament inference.",
            s["body"],
        ),
        p(
            "The most important result was methodological: more training steps did not reliably produce a stronger agent. Maze-only rewards could increase movement while reducing combat, and self-play could converge to repeated local behavior. Held-out official scoring, maze coverage, idle fraction, action coverage, visual playback, and peer matches were therefore all needed to choose a submission.",
            s["body"],
        ),
        p(
            "The selected pair performed well in the available peer test (five wins and one draw across six role-specific games), but its low and variable held-out scores show that the task is not solved. A future version should use recurrent memory or an explicit learned map, a larger opponent pool, and more held-out matches before checkpoint promotion.",
            s["body"],
        ),
        p("6. Reproducibility", s["h1"]),
        table(
            [
                ["Artifact", "Purpose"],
                ["src/official_environment.py", "Exact environment and training adapter"],
                ["src/train_official_ppo.py", "PPO training and checkpointing"],
                ["src/evaluate_official_ppo.py", "Unshaped role-specific evaluation"],
                ["src/export_official_dual_submission.py", "Two-role tournament export"],
                ["scripts/overnight_official_13h.sh", "Reproducible long-run curriculum"],
            ],
            [69 * mm, 101 * mm],
            s,
        ),
        Spacer(1, 7),
        p("References", s["h2"]),
        p(
            "Course repository: https://github.com/Hautamaki-lab/Summer-School-2026<br/>PettingZoo Combat Tank: https://pettingzoo.farama.org/environments/atari/combat_tank/<br/>Schulman et al. (2017), Proximal Policy Optimization Algorithms.",
            s["small"],
        ),
    ]

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
