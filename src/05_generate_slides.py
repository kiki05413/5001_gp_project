"""
DSC5001 Project: Generate 10-page PowerPoint slides
Title: Hong Kong AQHI High-Risk Hour Early Warning
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))

# Color scheme - Academic blue/white
COLOR_TITLE = RGBColor(0x1A, 0x23, 0x7E)
COLOR_ACCENT = RGBColor(0x0D, 0x47, 0xA1)
COLOR_TEXT = RGBColor(0x21, 0x21, 0x21)
COLOR_LIGHT_BG = RGBColor(0xF5, 0xF7, 0xFA)
COLOR_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
COLOR_RED = RGBColor(0xD3, 0x2F, 0x2F)
COLOR_GREEN = RGBColor(0x2E, 0x7D, 0x32)
COLOR_BLUE = RGBColor(0x15, 0x65, 0xC0)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

SLIDE_W = prs.slide_width
SLIDE_H = prs.slide_height


def add_textbox(slide, left, top, width, height, text, font_size=14, bold=False, 
                color=COLOR_TEXT, alignment=PP_ALIGN.LEFT, font_name='Calibri'):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_bullet_list(slide, left, top, width, height, items, font_size=13, 
                    color=COLOR_TEXT, spacing=Pt(6)):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        
        if isinstance(item, tuple):
            p.text = item[0]
            p.font.bold = item[1]
        else:
            p.text = item
            p.font.bold = False
        
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = 'Calibri'
        p.space_after = spacing
        p.level = 0
    
    return txBox


def add_title_bar(slide, title_text, subtitle_text=None):
    """Add a colored title bar at the top."""
    # Background bar
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(1.0)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = COLOR_ACCENT
    shape.line.fill.background()
    
    # Title text
    add_textbox(slide, Inches(0.5), Inches(0.15), Inches(12), Inches(0.5),
                title_text, font_size=24, bold=True, color=COLOR_WHITE,
                alignment=PP_ALIGN.LEFT)
    
    if subtitle_text:
        add_textbox(slide, Inches(0.5), Inches(0.65), Inches(12), Inches(0.3),
                    subtitle_text, font_size=12, color=RGBColor(0xBB, 0xDE, 0xFB),
                    alignment=PP_ALIGN.LEFT)


def add_image(slide, image_path, left, top, width=None, height=None):
    if width and height:
        return slide.shapes.add_picture(image_path, left, top, width, height)
    elif width:
        return slide.shapes.add_picture(image_path, left, top, width=width)
    elif height:
        return slide.shapes.add_picture(image_path, left, top, height=height)
    else:
        return slide.shapes.add_picture(image_path, left, top)


def add_footer(slide, page_num, total=10):
    add_textbox(slide, Inches(0.5), Inches(7.1), Inches(6), Inches(0.3),
                "DSC5001 Project | Hong Kong AQHI High-Risk Hour Early Warning",
                font_size=9, color=RGBColor(0x75, 0x75, 0x75))
    add_textbox(slide, Inches(11), Inches(7.1), Inches(2), Inches(0.3),
                f"{page_num} / {total}",
                font_size=9, color=RGBColor(0x75, 0x75, 0x75), alignment=PP_ALIGN.RIGHT)


# ============================================================
# Slide 1: Title Page
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout

# Full background
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
shape.fill.solid()
shape.fill.fore_color.rgb = COLOR_ACCENT
shape.line.fill.background()

# Top accent line
shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.5), SLIDE_W, Inches(0.06))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0x82, 0xB1, 0xFF)
shape.line.fill.background()

# Course name
add_textbox(slide, Inches(1), Inches(1.0), Inches(11), Inches(0.5),
            "DSC5001 Statistical Machine Learning", font_size=18, 
            color=RGBColor(0xBB, 0xDE, 0xFB), alignment=PP_ALIGN.CENTER)

# Project title
add_textbox(slide, Inches(1), Inches(2.8), Inches(11), Inches(1.0),
            "Hong Kong AQHI High-Risk Hour Early Warning:", font_size=28, 
            bold=True, color=COLOR_WHITE, alignment=PP_ALIGN.CENTER)

add_textbox(slide, Inches(1), Inches(3.8), Inches(11), Inches(0.6),
            "A Comparative Study of Classification Models on Hourly Monitoring Data",
            font_size=18, color=RGBColor(0x90, 0xCA, 0xF9), alignment=PP_ALIGN.CENTER)

# Team info
add_textbox(slide, Inches(1), Inches(5.2), Inches(11), Inches(0.4),
            "Team-XX", font_size=16, bold=True, color=COLOR_WHITE, alignment=PP_ALIGN.CENTER)

add_textbox(slide, Inches(1), Inches(5.6), Inches(11), Inches(0.4),
            "[Student Name 1] (UID)  |  [Student Name 2] (UID)  |  [Student Name 3] (UID)  |  [Student Name 4] (UID)",
            font_size=12, color=RGBColor(0xBB, 0xDE, 0xFB), alignment=PP_ALIGN.CENTER)

add_textbox(slide, Inches(1), Inches(6.2), Inches(11), Inches(0.4),
            "October 2025", font_size=14, color=RGBColor(0x90, 0xCA, 0xF9), alignment=PP_ALIGN.CENTER)


# ============================================================
# Slide 2: Research Question & Motivation
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Research Question & Motivation")

# Left column - Question
add_textbox(slide, Inches(0.5), Inches(1.3), Inches(6), Inches(0.4),
            "Core Question", font_size=16, bold=True, color=COLOR_ACCENT)

add_bullet_list(slide, Inches(0.5), Inches(1.8), Inches(6), Inches(3.5), [
    "Can we reliably predict whether next-hour AQHI will reach high-risk level (>=7) using past observations?",
    "How do different classifiers trade off between catching high-risk hours vs. controlling false alarms?",
    "Three models compared: Logistic Regression, Random Forest, XGBoost",
    ("Why classification, not regression?", True),
    "Public health advice is risk-level-based, not numeric -> classification matches real application",
], font_size=12)

# Right column - Motivation
add_textbox(slide, Inches(7), Inches(1.3), Inches(6), Inches(0.4),
            "Motivation", font_size=16, bold=True, color=COLOR_ACCENT)

add_bullet_list(slide, Inches(7), Inches(1.8), Inches(6), Inches(3.5), [
    "EPD publishes AQHI hourly but forecasts only 3x/day (00:30, 10:30, 16:30)",
    "Hourly-level early warning has room for improvement during rapid pollution spikes",
    "AQHI time series has temporal dependency + severe class imbalance (<2% high-risk)",
    ("Scientific value: not chasing best accuracy, but analyzing model trade-offs", True),
    "Missed alerts (FN) vs. false alarms (FP) have different real-world costs",
], font_size=12)

add_footer(slide, 2)


# ============================================================
# Slide 3: Data Sources & Feature Engineering
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Data Sources & Feature Engineering")

# Left - Data
add_textbox(slide, Inches(0.5), Inches(1.3), Inches(6), Inches(0.4),
            "Data Sources", font_size=16, bold=True, color=COLOR_ACCENT)

add_bullet_list(slide, Inches(0.5), Inches(1.8), Inches(6), Inches(4), [
    ("AQHI Hourly Data (EPD / data.gov.hk)", True),
    "18 stations: 15 general + 3 roadside (Causeway Bay, Central, Mong Kok)",
    "Period: Jan 2021 - Dec 2024 (48 months, avoiding 2025 methodology change)",
    "Total: 617,799 samples after preprocessing",
    ("Weather Data (HKO / data.gov.hk)", True),
    "Daily mean temperature, relative humidity, rainfall, wet bulb temp",
    "Merged by date with AQHI records",
], font_size=11)

# Right - Features
add_textbox(slide, Inches(7), Inches(1.3), Inches(6), Inches(0.4),
            "Feature Engineering (27 features)", font_size=16, bold=True, color=COLOR_ACCENT)

add_bullet_list(slide, Inches(7), Inches(1.8), Inches(6), Inches(4), [
    ("Lagged AQHI: lag-1, 2, 3, 6, 12, 24 hours", True),
    ("Rolling stats: 3h/6h mean, 6h std", True),
    ("AQHI differences: 1-step and 3-step", True),
    ("Time encoding: hour, day-of-week, month (sin/cos)", True),
    ("Weather: temperature, humidity, rainfall, wet bulb", True),
    ("Station type: roadside vs. general (binary)", True),
    ("Target: next-hour AQHI >= 7 -> binary (1.60% positive)", True),
], font_size=11)

add_footer(slide, 3)


# ============================================================
# Slide 4: Methodology
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Methodology")

# Left - Models
add_textbox(slide, Inches(0.5), Inches(1.3), Inches(6), Inches(0.4),
            "Three Classification Models", font_size=16, bold=True, color=COLOR_ACCENT)

add_bullet_list(slide, Inches(0.5), Inches(1.8), Inches(6), Inches(3), [
    ("1. Logistic Regression (baseline)", True),
    "  L2 regularization, class_weight='balanced'",
    ("2. Random Forest", True),
    "  200 trees, max_depth=15, class_weight='balanced'",
    ("3. XGBoost (Gradient Boosting)", True),
    "  300 rounds, max_depth=6, scale_pos_weight",
], font_size=12)

# Right - Evaluation Strategy
add_textbox(slide, Inches(7), Inches(1.3), Inches(6), Inches(0.4),
            "Evaluation Strategy", font_size=16, bold=True, color=COLOR_ACCENT)

add_bullet_list(slide, Inches(7), Inches(1.8), Inches(6), Inches(4), [
    ("TimeSeriesSplit CV (5 folds)", True),
    "  Prevents future data leakage",
    "  Uses only past data to predict future",
    ("Class imbalance handling:", True),
    "  Threshold optimization via PR curve",
    "  No SMOTE (preserves temporal structure)",
    ("Key metrics:", True),
    "  Recall (catch high-risk hours)",
    "  Precision (control false alarms)",
    "  F1, PR-AUC, Confusion Matrix",
    ("Cost analysis: FN cost = 5x FP cost", True),
], font_size=11)

add_footer(slide, 4)


# ============================================================
# Slide 5: Data Exploration
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Data Exploration: AQHI Distribution & Patterns")

# Left image - distribution
img_path = os.path.join(FIGURES_DIR, '01_aqhi_distribution.png')
if os.path.exists(img_path):
    add_image(slide, img_path, Inches(0.3), Inches(1.2), width=Inches(6.5))

# Right image - hourly pattern
img_path = os.path.join(FIGURES_DIR, '08_hourly_pattern.png')
if os.path.exists(img_path):
    add_image(slide, img_path, Inches(6.9), Inches(1.2), width=Inches(6.2))

# Key findings at bottom
add_textbox(slide, Inches(0.5), Inches(6.3), Inches(12), Inches(0.6),
    "Key findings: AQHI is right-skewed with only 1.60% high-risk (>=7). Roadside stations show consistently higher AQHI than general stations. "
    "High-risk hours peak during afternoon (12:00-19:00) across both station types.",
    font_size=11, color=COLOR_TEXT)

add_footer(slide, 5)


# ============================================================
# Slide 6: Cross-Validation Results
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Cross-Validation Results (5-Fold TimeSeriesSplit)")

# Left image - CV comparison boxplot
img_path = os.path.join(FIGURES_DIR, '04_cv_comparison.png')
if os.path.exists(img_path):
    add_image(slide, img_path, Inches(0.3), Inches(1.2), width=Inches(7.5))

# Right - Summary table
add_textbox(slide, Inches(8.2), Inches(1.3), Inches(5), Inches(0.4),
            "CV Summary (mean +/- std)", font_size=14, bold=True, color=COLOR_ACCENT)

# Table data
table_data = [
    ["Model", "Prec.", "Recall", "F1", "PR-AUC"],
    ["Log. Reg.", "0.566\n+/-0.118", "0.686\n+/-0.051", "0.616\n+/-0.090", "0.679\n+/-0.101"],
    ["Rand. Forest", "0.580\n+/-0.111", "0.693\n+/-0.039", "0.627\n+/-0.080", "0.663\n+/-0.111"],
    ["XGBoost", "0.481\n+/-0.119", "0.663\n+/-0.015", "0.550\n+/-0.088", "0.580\n+/-0.104"],
]

rows = len(table_data)
cols = len(table_data[0])
table = slide.shapes.add_table(rows, cols, Inches(8.2), Inches(1.8), Inches(5), Inches(3)).table

for i, row_data in enumerate(table_data):
    for j, cell_text in enumerate(row_data):
        cell = table.cell(i, j)
        cell.text = cell_text
        for paragraph in cell.text_frame.paragraphs:
            paragraph.font.size = Pt(10) if i > 0 else Pt(11)
            paragraph.font.bold = (i == 0)
            paragraph.alignment = PP_ALIGN.CENTER
            if i == 0:
                paragraph.font.color.rgb = COLOR_WHITE
        
        if i == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_ACCENT
        elif i % 2 == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = COLOR_LIGHT_BG

add_textbox(slide, Inches(8.2), Inches(5.5), Inches(5), Inches(1),
    "Logistic Regression and Random Forest perform comparably. XGBoost shows lower precision but more stable recall across folds.",
    font_size=10, color=COLOR_TEXT)

add_footer(slide, 6)


# ============================================================
# Slide 7: PR Curves & Confusion Matrices
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Test Set Results: PR Curves & Confusion Matrices")

# Left - PR curves
img_path = os.path.join(FIGURES_DIR, '02_pr_curves.png')
if os.path.exists(img_path):
    add_image(slide, img_path, Inches(0.3), Inches(1.2), width=Inches(6.3))

# Right - Confusion matrices
img_path = os.path.join(FIGURES_DIR, '03_confusion_matrices.png')
if os.path.exists(img_path):
    add_image(slide, img_path, Inches(6.8), Inches(1.2), width=Inches(6.5))

add_footer(slide, 7)


# ============================================================
# Slide 8: Feature Importance
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Feature Importance Analysis")

# Top left - RF & XGBoost importance
img_path = os.path.join(FIGURES_DIR, '05_feature_importance.png')
if os.path.exists(img_path):
    add_image(slide, img_path, Inches(0.3), Inches(1.2), width=Inches(7))

# Bottom left - LR coefficients
img_path = os.path.join(FIGURES_DIR, '06_lr_coefficients.png')
if os.path.exists(img_path):
    add_image(slide, img_path, Inches(0.3), Inches(4.0), width=Inches(7))

# Right - Key findings
add_textbox(slide, Inches(7.8), Inches(1.3), Inches(5.2), Inches(0.4),
            "Key Findings", font_size=16, bold=True, color=COLOR_ACCENT)

add_bullet_list(slide, Inches(7.8), Inches(1.8), Inches(5.2), Inches(5), [
    ("AQHI_lag_1 dominates all models", True),
    "  Most recent observation is strongest predictor",
    ("Rolling mean (3h) is 2nd most important", True),
    "  Recent trend captures momentum",
    ("Rolling std (6h) matters for RF", True),
    "  Volatility signals impending risk",
    ("Hour-of-day patterns are significant", True),
    "  Afternoon hours (12-19) show higher risk",
    ("Temperature has positive coefficient in LR", True),
    "  Higher temp -> more high-risk hours",
    ("Rainfall has negative coefficient", True),
    "  Rain washes pollutants from air",
], font_size=11)

add_footer(slide, 8)


# ============================================================
# Slide 9: Cost-Benefit Analysis & Threshold
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Cost-Benefit Analysis & Threshold Optimization")

# Left - Cost benefit
img_path = os.path.join(FIGURES_DIR, '07_cost_benefit.png')
if os.path.exists(img_path):
    add_image(slide, img_path, Inches(0.3), Inches(1.2), width=Inches(6.8))

# Right - Threshold sensitivity
img_path = os.path.join(FIGURES_DIR, '10_threshold_sensitivity.png')
if os.path.exists(img_path):
    add_image(slide, img_path, Inches(7.3), Inches(1.2), width=Inches(5.8))

# Bottom findings
add_textbox(slide, Inches(0.5), Inches(6.3), Inches(12.5), Inches(0.6),
    "Assumption: missing a high-risk alert (FN) costs 5x a false alarm (FP). Random Forest achieves the highest cost reduction (61.3%), "
    "with the fewest missed alerts (448) despite more false alarms. Optimal thresholds are model-specific: LR=0.978, RF=0.893, XGB=0.748.",
    font_size=10, color=COLOR_TEXT)

add_footer(slide, 9)


# ============================================================
# Slide 10: Conclusions & Limitations
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_title_bar(slide, "Conclusions & Limitations")

# Left - Conclusions
add_textbox(slide, Inches(0.5), Inches(1.3), Inches(6), Inches(0.4),
            "Conclusions", font_size=16, bold=True, color=COLOR_GREEN)

add_bullet_list(slide, Inches(0.5), Inches(1.8), Inches(6), Inches(4.5), [
    ("Classification framework is viable for AQHI early warning", True),
    "  Binary prediction matches public health advice structure",
    ("Random Forest: best recall (73.8%) & cost reduction (61.3%)", True),
    "  Recommended when missing high-risk hours is most costly",
    ("Logistic Regression: best precision (60.8%) & highest PR-AUC", True),
    "  Recommended when false alarms must be minimized",
    ("XGBoost: balanced but lower overall performance", True),
    "  May benefit from more hyperparameter tuning",
    ("Key predictor: AQHI_lag_1 (most recent observation)", True),
    "  Simple lagged features outperform meteorological variables",
], font_size=11)

# Right - Limitations & Future Work
add_textbox(slide, Inches(7), Inches(1.3), Inches(6), Inches(0.4),
            "Limitations & Future Work", font_size=16, bold=True, color=COLOR_RED)

add_bullet_list(slide, Inches(7), Inches(1.8), Inches(6), Inches(4.5), [
    ("Weather data is daily, not hourly", True),
    "  Limits temporal resolution of meteorological features",
    ("Station spatial mismatch", True),
    "  EPD stations and HKO weather stations don't co-locate",
    ("AQHI methodology changed in March 2025", True),
    "  Used 2021-2024 data to avoid distribution shift",
    ("Class imbalance is severe (1.60%)", True),
    "  Even with weighting, precision remains moderate",
    ("Future: hourly weather data, multi-station spatial models", True),
    "  Deep learning (LSTM/Transformer) for sequence prediction",
], font_size=11)

add_footer(slide, 10)


# ============================================================
# Save
# ============================================================
output_path = os.path.join(OUTPUT_DIR, 'DSC5001_Project_Slides.pptx')
prs.save(output_path)
print(f"PPTX saved to: {output_path}")
print(f"Total slides: {len(prs.slides)}")
