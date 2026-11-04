"""Generate a single-page research poster from the latest AQHI model results."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "model_results.json")
HOURLY_FIGURE_PATH = os.path.join(FIGURES_DIR, "08_hourly_pattern.png")
POSTER_PATH = os.path.join(OUTPUT_DIR, "DSC5001_Project_Poster.pdf")

NAVY = "#102A43"
BLUE = "#176B87"
TEAL = "#26A69A"
ORANGE = "#E98B35"
RED = "#D45D5D"
INK = "#203040"
MUTED = "#637587"
PALE = "#F2F6F8"
LINE = "#DCE5EA"
WHITE = "#FFFFFF"

PAGE_W = 36
PAGE_H = 24
MARGIN = 0.55


def add_panel(fig, x, top, width, height, title, accent=BLUE):
    bottom = top - height
    panel = FancyBboxPatch(
        (x / PAGE_W, bottom / PAGE_H),
        width / PAGE_W,
        height / PAGE_H,
        boxstyle="round,pad=0.003,rounding_size=0.005",
        transform=fig.transFigure,
        linewidth=1.1,
        edgecolor=LINE,
        facecolor=WHITE,
        zorder=0,
    )
    fig.add_artist(panel)
    fig.add_artist(
        Rectangle(
            ((x + 0.02) / PAGE_W, (bottom + 0.18) / PAGE_H),
            0.08 / PAGE_W,
            (height - 0.36) / PAGE_H,
            transform=fig.transFigure,
            facecolor=accent,
            edgecolor="none",
            zorder=1,
        )
    )
    fig.text(
        (x + 0.32) / PAGE_W,
        (top - 0.43) / PAGE_H,
        title.upper(),
        color=accent,
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    return bottom


def add_text(fig, x, y, text, size=18, color=INK, weight="normal", **kwargs):
    fig.text(
        x / PAGE_W,
        y / PAGE_H,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        va=kwargs.pop("va", "top"),
        ha=kwargs.pop("ha", "left"),
        linespacing=kwargs.pop("linespacing", 1.35),
        **kwargs,
    )


def make_chart_axes(fig, x, y, width, height):
    return fig.add_axes([x / PAGE_W, y / PAGE_H, width / PAGE_W, height / PAGE_H])


def build_poster():
    if not os.path.isfile(RESULTS_PATH):
        raise FileNotFoundError(f"Latest model results not found: {RESULTS_PATH}")
    if not os.path.isfile(HOURLY_FIGURE_PATH):
        raise FileNotFoundError(f"Hourly pattern figure not found: {HOURLY_FIGURE_PATH}")

    with open(RESULTS_PATH, "r", encoding="utf-8") as results_file:
        results = json.load(results_file)

    test_results = results["test_results"]
    cv_summary = results["cv_summary"]
    rf_features = results["feature_importance"]["random_forest"][:5]

    model_order = [
        ("Logistic Regression", "logistic_regression", BLUE),
        ("Random Forest", "random_forest", TEAL),
        ("XGBoost", "xgboost", ORANGE),
    ]

    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=PALE)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.add_artist(
        Rectangle(
            (0, 0.875),
            1,
            0.125,
            transform=fig.transFigure,
            facecolor=NAVY,
            edgecolor="none",
            zorder=0,
        )
    )
    fig.add_artist(
        Rectangle(
            (0, 0.875),
            0.012,
            0.125,
            transform=fig.transFigure,
            facecolor=TEAL,
            edgecolor="none",
            zorder=1,
        )
    )
    add_text(fig, 1.26, 23.472, "DSC5001  /  STATISTICAL MACHINE LEARNING", 14, "#A9CFD9", "bold")
    add_text(fig, 1.26, 22.800, "EARLY WARNING FOR HIGH-RISK AIR QUALITY HOURS", 34, WHITE, "bold")
    add_text(
        fig,
        1.296,
        21.888,
        "A time-aware comparison of classification models for Hong Kong's Air Quality Health Index",
        20,
        "#D8E6EC",
    )
    add_text(fig, 34.74, 23.328, "HONG KONG  |  2021–2024", 15, WHITE, "bold", ha="right")
    add_text(fig, 34.74, 22.44, "EPD hourly AQHI  +  HKO daily weather", 14, "#C6D8E0", ha="right")

    rf_test = test_results["random_forest"]["optimal_threshold"]
    rf_cost = test_results["random_forest"]["cost_analysis"]
    rf_cv = cv_summary["random_forest"]["pr_auc"]["mean"]
    kpis = [
        ("73.8%", "Random Forest recall", "at threshold 0.893"),
        (f"{rf_cost['cost_reduction_pct']:.1f}%", "Lower alert cost", "vs. no-model baseline"),
        (f"{rf_test['pr_auc']:.3f}", "Test PR-AUC", f"5-fold CV mean: {rf_cv:.3f}"),
    ]
    kpi_xs = [0.26, 0.50, 0.74]
    for (value, label, detail), center in zip(kpis, kpi_xs):
        fig.add_artist(
            FancyBboxPatch(
                (center - 0.105, 0.828),
                0.21,
                0.043,
                boxstyle="round,pad=0.008,rounding_size=0.009",
                transform=fig.transFigure,
                facecolor=WHITE,
                edgecolor=LINE,
                linewidth=0.8,
                zorder=0,
            )
        )
        add_text(fig, (center - 0.09) * PAGE_W, 0.858 * PAGE_H, value, 23, TEAL, "bold")
        add_text(fig, (center - 0.025) * PAGE_W, 0.856 * PAGE_H, label, 15, INK, "bold")
        add_text(fig, (center - 0.025) * PAGE_W, 0.836 * PAGE_H, detail, 12, MUTED)

    left_x, left_w = 0.55, 10.75
    mid_x, mid_w = 11.62, 12.38
    right_x, right_w = 24.32, 11.13
    body_top = 19.55

    add_panel(fig, left_x, body_top, left_w, 1.85, "Research question", BLUE)
    add_text(
        fig,
        left_x + 0.34,
        body_top - 0.82,
        "Can recent station readings and AQHI trends warn us\none hour before the AQHI reaches 7?",
        19,
        INK,
        "bold",
        wrap=True,
    )
    add_text(fig, left_x + 0.34, body_top - 1.48, "A risk-level alert is more actionable than a point forecast.", 16, MUTED)

    add_panel(fig, left_x, 17.48, left_w, 4.00, "Data at a glance", TEAL)
    data_items = [
        ("617,799", "hourly station records"),
        ("18", "monitoring stations"),
        ("27", "engineered features"),
    ]
    for i, (value, label) in enumerate(data_items):
        x = left_x + 0.42 + i * 3.42
        add_text(fig, x, 16.63, value, 25, TEAL, "bold")
        add_text(fig, x, 16.15, label, 14, MUTED)
    add_text(
        fig,
        left_x + 0.42,
        15.58,
        "Target: next-hour AQHI ≥ 7  |  Positive class: 1.60%",
        17,
        INK,
        "bold",
    )
    add_text(
        fig,
        left_x + 0.42,
        15.03,
        "AQHI history (lags, rolling means, volatility, changes) + time-of-day,\n"
        "station type, and daily temperature, humidity, rainfall and wet-bulb data.",
        15,
        MUTED,
    )

    add_panel(fig, left_x, 13.18, left_w, 5.15, "Evaluation design", BLUE)
    steps = [
        ("01", "Chronological split", "Final 20% held out as the test period."),
        ("02", "TimeSeriesSplit", "Five expanding-window folds; no future-to-past leakage."),
        ("03", "Three classifiers", "Logistic Regression, Random Forest and XGBoost."),
        ("04", "Risk-aware scoring", "Precision, recall, F1 and PR-AUC; FN cost = 5× FP."),
    ]
    for i, (number, title, detail) in enumerate(steps):
        y = 12.05 - i * 1.07
        add_text(fig, left_x + 0.38, y, number, 17, TEAL, "bold")
        add_text(fig, left_x + 1.18, y, title, 17, INK, "bold")
        add_text(fig, left_x + 1.18, y - 0.37, detail, 15, MUTED)

    add_panel(fig, left_x, 7.78, left_w, 6.35, "Sources & study boundaries", ORANGE)
    add_text(
        fig,
        left_x + 0.38,
        6.88,
        "DATA SOURCES",
        14,
        ORANGE,
        "bold",
    )
    add_text(
        fig,
        left_x + 0.38,
        6.48,
        "Hong Kong EPD hourly AQHI (2021–2024); Hong Kong Observatory daily\n"
        "weather observations, merged by date.",
        15,
        INK,
    )
    add_text(fig, left_x + 0.38, 5.55, "INTERPRETATION", 14, ORANGE, "bold")
    add_text(
        fig,
        left_x + 0.38,
        5.15,
        "A strong recent-AQHI signal supports short-horizon warning, but does\n"
        "not establish causal effects of weather or station characteristics.",
        15,
        INK,
    )
    add_text(fig, left_x + 0.38, 4.22, "LIMITATIONS", 14, ORANGE, "bold")
    add_text(
        fig,
        left_x + 0.38,
        3.82,
        "Weather inputs are daily, not hourly; station locations do not align\n"
        "perfectly. AQHI methodology changed in 2025, so that period is excluded.",
        15,
        INK,
    )

    add_panel(fig, mid_x, body_top, mid_w, 6.65, "Held-out test results", TEAL)
    table_left = mid_x + 0.34
    table_top = body_top - 1.14
    col_widths = [2.67, 1.72, 1.65, 1.72, 1.55, 1.46]
    headers = ["MODEL", "THRESH.", "PR-AUC", "PRECISION", "RECALL", "F1"]
    xpos = table_left
    for header, width in zip(headers, col_widths):
        add_text(fig, xpos, table_top, header, 11, MUTED, "bold")
        xpos += width
    for i, (name, key, color) in enumerate(model_order):
        row_top = table_top - 0.68 - i * 0.95
        metric = test_results[key]["optimal_threshold"]
        row_bottom = row_top - 0.68
        fig.add_artist(
            FancyBboxPatch(
                (table_left - 0.06, row_bottom),
                mid_w - 0.56,
                0.78,
                boxstyle="round,pad=0.005,rounding_size=0.05",
                transform=fig.transFigure,
                facecolor="#EAF5F3" if key == "random_forest" else "#F8FAFB",
                edgecolor="none",
                zorder=0,
            )
        )
        values = [
            name,
            f"{metric['threshold']:.3f}",
            f"{metric['pr_auc']:.3f}",
            f"{metric['precision']:.3f}",
            f"{metric['recall']:.3f}",
            f"{metric['f1']:.3f}",
        ]
        xpos = table_left
        for j, (value, width) in enumerate(zip(values, col_widths)):
            add_text(
                fig,
                xpos,
                row_top,
                value,
                14 if j == 0 else 15,
                color if j == 0 else INK,
                "bold" if j == 0 or key == "random_forest" else "normal",
            )
            xpos += width
    add_text(
        fig,
        table_left,
        15.10,
        "RF leads test recall and PR-AUC; LR gives the highest precision.",
        15,
        INK,
        "bold",
    )
    add_text(
        fig,
        table_left,
        14.62,
        "Thresholds selected on test probabilities (min recall 0.65; maximize F1).",
        12,
        MUTED,
    )
    add_text(
        fig,
        table_left,
        14.22,
        "Treat threshold-dependent test metrics as exploratory; tune on validation data before deployment.",
        12,
        RED,
        "bold",
    )

    add_panel(fig, mid_x, 12.60, mid_w, 3.42, "Five-fold time-series cross-validation", BLUE)
    cv_ax = make_chart_axes(fig, mid_x + 1.45, 9.55, mid_w - 2.15, 1.40)
    cv_values = [cv_summary[key]["pr_auc"]["mean"] for _, key, _ in model_order]
    cv_errors = [cv_summary[key]["pr_auc"]["std"] for _, key, _ in model_order]
    cv_colors = [color for _, _, color in model_order]
    cv_ax.barh(range(3), cv_values, xerr=cv_errors, color=cv_colors, alpha=0.9, capsize=5)
    cv_ax.set_yticks(range(3), [name for name, _, _ in model_order], fontsize=11, color=INK)
    cv_ax.invert_yaxis()
    cv_ax.set_xlim(0, 0.9)
    cv_ax.set_xlabel("Mean PR-AUC ± standard deviation", fontsize=11, color=MUTED)
    cv_ax.tick_params(axis="x", labelsize=10, colors=MUTED)
    cv_ax.grid(axis="x", color=LINE, linewidth=0.8)
    cv_ax.set_axisbelow(True)
    for spine in cv_ax.spines.values():
        spine.set_visible(False)
    add_text(fig, mid_x + 0.36, 11.47, "Logistic Regression has the highest mean CV PR-AUC.", 12, INK, "bold")

    add_panel(fig, mid_x, 8.83, mid_w, 7.10, "Alert cost under FN = 5 × FP", ORANGE)
    cost_ax = make_chart_axes(fig, mid_x + 2.25, 2.75, mid_w - 3.05, 4.45)
    baseline = test_results["random_forest"]["cost_analysis"]["baseline_cost"]
    costs = [baseline]
    names = ["No-model\nbaseline"]
    colors = [MUTED]
    reductions = [0]
    for name, key, color in model_order:
        cost_data = test_results[key]["cost_analysis"]
        costs.append(cost_data["total_cost"])
        names.append(name.replace(" ", "\n", 1))
        colors.append(color)
        reductions.append(cost_data["cost_reduction_pct"])
    bars = cost_ax.barh(range(4), costs, color=colors, height=0.58)
    cost_ax.invert_yaxis()
    cost_ax.set_yticks(range(4), names, fontsize=11, color=INK)
    cost_ax.set_xlim(0, baseline * 1.22)
    cost_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    cost_ax.tick_params(axis="x", labelsize=10, colors=MUTED)
    cost_ax.set_xlabel("Weighted cost units (lower is better)", fontsize=11, color=MUTED)
    cost_ax.grid(axis="x", color=LINE, linewidth=0.8)
    cost_ax.set_axisbelow(True)
    for spine in cost_ax.spines.values():
        spine.set_visible(False)
    for i, (bar, value, reduction) in enumerate(zip(bars, costs, reductions)):
        label = f"{value:,}" if i == 0 else f"{value:,}   (−{reduction:.1f}%)"
        cost_ax.text(value + baseline * 0.012, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=11, color=INK)
    add_text(fig, mid_x + 0.36, 2.28, "Random Forest: fewest missed alerts (448) and lowest total cost.", 14, INK, "bold")
    add_text(fig, mid_x + 0.36, 1.88, "Cost assumption: each missed high-risk hour costs five false alarms.", 12, MUTED)

    add_panel(fig, right_x, body_top, right_w, 6.25, "What drives the warning?", BLUE)
    add_text(fig, right_x + 0.38, 18.52, "Recent AQHI levels dominate the model's feature importance.", 13, INK, "bold")
    feature_ax = make_chart_axes(fig, right_x + 0.68, 13.50, right_w - 1.18, 4.50)
    feature_names = [item["feature"].replace("AQHI_", "").replace("_", " ") for item in rf_features]
    feature_values = [item["importance"] for item in rf_features]
    feature_ax.barh(range(len(feature_values)), feature_values, color=[TEAL, BLUE, "#56A6B8", "#72B5C0", ORANGE], height=0.62)
    feature_ax.set_yticks(range(len(feature_names)), feature_names, fontsize=11, color=INK)
    feature_ax.invert_yaxis()
    feature_ax.set_xlim(0, max(feature_values) * 1.27)
    feature_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    feature_ax.tick_params(axis="x", labelsize=10, colors=MUTED)
    feature_ax.set_xlabel("Random Forest feature importance", fontsize=11, color=MUTED)
    feature_ax.grid(axis="x", color=LINE, linewidth=0.8)
    feature_ax.set_axisbelow(True)
    for spine in feature_ax.spines.values():
        spine.set_visible(False)
    for i, value in enumerate(feature_values):
        feature_ax.text(value + max(feature_values) * 0.025, i, f"{value:.3f}", va="center", fontsize=10, color=INK)
    add_panel(fig, right_x, 13.00, right_w, 4.05, "When is high risk most common?", TEAL)
    hourly_image = Image.open(HOURLY_FIGURE_PATH)
    hourly_ax = make_chart_axes(fig, right_x + 0.34, 9.08, right_w - 0.68, 3.02)
    hourly_ax.imshow(hourly_image)
    hourly_ax.axis("off")

    add_panel(fig, right_x, 8.65, right_w, 2.70, "Practical takeaway", TEAL)
    add_text(
        fig,
        right_x + 0.38,
        7.72,
        "Choose Random Forest when missed alerts matter most.\n"
        "Choose Logistic Regression when fewer false alarms\n"
        "and higher precision are the priority.",
        16,
        INK,
        "bold",
    )

    add_panel(fig, right_x, 5.65, right_w, 4.55, "Caveat before deployment", RED)
    add_text(
        fig,
        right_x + 0.38,
        4.75,
        "The reported alert thresholds were selected using the\n"
        "held-out test probabilities. This makes threshold-based\n"
        "test scores exploratory, not unbiased final estimates.\n\n"
        "Select thresholds on a validation period, then evaluate\n"
        "once on a separate untouched test period.",
        15,
        INK,
    )

    fig.add_artist(
        Rectangle(
            (0, 0),
            1,
            0.038,
            transform=fig.transFigure,
            facecolor=NAVY,
            edgecolor="none",
            zorder=0,
        )
    )
    add_text(
        fig,
        MARGIN,
        0.022 * PAGE_H,
        "SOURCES: Hong Kong EPD AQHI  |  Hong Kong Observatory daily weather  |  Model results: 5-fold time-series CV + chronological holdout",
        11,
        "#DCE5EA",
        "normal",
        va="center",
    )
    add_text(
        fig,
        0.965 * PAGE_W,
        0.022 * PAGE_H,
        "NEXT-HOUR AQHI ≥ 7  •  RESEARCH PROTOTYPE",
        11,
        "#DCE5EA",
        "bold",
        ha="right",
        va="center",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(POSTER_PATH, format="pdf", facecolor=PALE)
    plt.close(fig)
    print(f"Poster saved to: {POSTER_PATH}")
    print("Page size: 36 × 24 in | Pages: 1")


if __name__ == "__main__":
    build_poster()
