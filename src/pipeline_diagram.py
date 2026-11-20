"""
Research Pipeline Architecture & Flowchart Generator
Generates Figure 1: Comprehensive ML & NLP Research Architecture for ESG & Financial Risk.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_pipeline_flowchart(output_path: str):
    """
    Draws a publication-quality end-to-end research methodology diagram:
    Stage 1: Multi-Source Empirical Data Panels (Dataset 1, Dataset 2, Harmonized Panel)
    Stage 2: Strict Temporal Partitioning (Train, Conformal Calibration, Out-of-Time Test)
    Stage 3: 12-Model Multi-Paradigm Modeling Suite (Tabular, Uncertainty, Transformers)
    Stage 4: Empirical Research Validation & Themes (Uncertainty, TreeSHAP, Distress, Greenwashing)
    """
    fig, ax = plt.subplots(figsize=(20, 12), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Clean Academic Palette
    c_data = "#f4f8fb"
    c_data_border = "#1f618d"
    c_split = "#fefcf4"
    c_split_border = "#d68910"
    c_model = "#f4faf6"
    c_model_border = "#229954"
    c_eval = "#fdf5f5"
    c_eval_border = "#b03a2e"

    def draw_card(x, y, w, h, bg, border, badge, title, sections):
        # Main container
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.2",
            facecolor=bg, edgecolor=border, linewidth=2.2, zorder=2
        )
        ax.add_patch(rect)

        # Top Badge
        badge_w = min(w * 0.72, 23)
        badge_rect = patches.FancyBboxPatch(
            (x + 2, y + h - 4.2), badge_w, 3.8,
            boxstyle="round,pad=0.2,rounding_size=0.6",
            facecolor=border, edgecolor=border, zorder=3
        )
        ax.add_patch(badge_rect)
        ax.text(x + 2 + badge_w / 2, y + h - 2.3, badge, 
                ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", zorder=4)

        # Section Title
        ax.text(x + w / 2, y + h - 7.5, title, 
                ha="center", va="center", color="#1c2833", fontsize=11.5, fontweight="bold", zorder=3)

        curr_y = y + h - 12.0
        for sec_header, lines in sections:
            if sec_header:
                ax.text(x + 2.5, curr_y, sec_header, 
                        ha="left", va="center", color=border, fontsize=9.2, fontweight="bold", zorder=3)
                curr_y -= 3.0
            for line in lines:
                ax.text(x + 4.2, curr_y, line, 
                        ha="left", va="center", color="#2c3e50", fontsize=8.4, zorder=3)
                curr_y -= 2.6
            curr_y -= 1.2

    # Title & Authors Header
    ax.text(50, 97.2, "End-to-End Machine Learning and NLP Framework for ESG Assessment & Financial Risk", 
            ha="center", va="center", fontsize=16.5, fontweight="bold", color="#17202a")
    ax.text(50, 94.2, "Rajnish Singh (@givemehat) · Medhavie Gupta · Katyayani Jaiswal (Gautam Buddha Univ.) | Guide: Prof. Valentina Lagasio (Sapienza Univ. of Rome)",
            ha="center", va="center", fontsize=10.5, fontstyle="italic", color="#566573")

    # Stage 1: Data Ingestion (Left Column)
    s1_sections = [
        ("Dataset 1: Global Multi-Agency Panel (12,000 Obs)", [
            "• 1,500 Global Corporations (2018–2025)",
            "• Orbis Financials: Assets, Leverage, ROA, Altman Z",
            "• Tri-Agency ESG Ratings (Refinitiv, MSCI, Sust.)",
            "• Rating Disagreement & Pillar Scores (E, S, G)",
            "• CSR Disclosures & Sentiment Polarity"
        ]),
        ("Dataset 2: UK Green Finance Panel (5,600 Obs)", [
            "• 800 UK LSE / FTSE 350 Firms (2019–2025)",
            "• Statutory SECR Audited Carbon Reductions",
            "• Green Bond Credit Spreads (45–650 bps)",
            "• Green CapEx Ratios & Statutory TCFD Compliance",
            "• Forward-Looking Statement (FLS) Vagueness"
        ]),
        ("Harmonized Pooled Dataset (17,600 Obs)", [
            "• Unified Standardized Dimensionless Constructs",
            "• Cross-Regime Empirical Benchmarking"
        ])
    ]
    draw_card(2.5, 6, 29, 84, c_data, c_data_border, "STAGE 1: DATA", "Multi-Source Empirical Panels", s1_sections)

    # Stage 2: Temporal Split (Top Center)
    s2_sections = [
        ("Strict Temporal Partitioning (Zero Lookahead)", [
            "• In-Sample Training Window: 2018–2022 (7,500 obs)",
            "• Conformal Calibration Slice: 2022 Holdout",
            "• Strict Out-of-Time Test Evaluation: 2023–2025",
            "• Zero Future Data Leakage Guarantee",
            "• Cross-Market Harmonization Standardization"
        ])
    ]
    draw_card(34.5, 53, 31, 37, c_split, c_split_border, "STAGE 2: SPLIT", "Temporal Partitioning & Calibration", s2_sections)

    # Stage 3: Core Modeling Engine (Bottom Center)
    s3_sections = [
        ("Tabular ML Suite (7 Architectures)", [
            "• M01: Ridge Regression (L2 Shrinkage OLS Baseline)",
            "• M02: Logistic Regression (L2 Default Classifier)",
            "• M03: Random Forest | M04: Extra Trees Ensemble",
            "• M05: GBM | M06: XGBoost | M07: LightGBM",
            "• M08: Deep Tabular Multi-Layer Perceptron (MLP)"
        ]),
        ("Uncertainty Quantification & Transformers (5 Architectures)", [
            "• M09: Quantile Gradient Boosted Trees (p10, p50, p90)",
            "• M10: Inductive Split Conformal Prediction (90% PICP)",
            "• M11: FinBERT Contextual Transformer Encoder",
            "• M12: Dual-Channel Transformer Discrepancy Network (RDI)"
        ])
    ]
    draw_card(34.5, 6, 31, 44, c_model, c_model_border, "STAGE 3: MODELS", "12-Model Multi-Paradigm Suite", s3_sections)

    # Stage 4: Research Validation (Right Column)
    s4_sections = [
        ("Theme A: Uncertainty & Guarantees", [
            "• RQ1: Conformal Prediction (86.7% empirical coverage)",
            "• Distribution-free finite-sample confidence intervals"
        ]),
        ("Theme B: Explainability & Temporal Stability", [
            "• RQ7: TreeSHAP Attribution Stability (rho = 0.66, Sim = 0.99)",
            "• Solvency dominance on D1 vs Carbon dominance on D2"
        ]),
        ("Theme C: Incremental Financial Distress Forecasting", [
            "• RQ2, RQ5, RQ6: Stepwise Testing (M1 to M5 Models)",
            "• Deep Tabular MLP wins on Global Distress (AUC = 0.5861)",
            "• Extra Trees wins on UK Credit Spreads (R2 = 0.9254)"
        ]),
        ("Theme D: Greenwashing NLP & Domain Transfer", [
            "• RQ3, RQ4: Dual-Channel Discrepancy (99.82% ROC-AUC)",
            "• Reporting Discrepancy Index (RDI > +0.85 greenwashing)",
            "• Cross-market domain transfer penalty (0.5474 -> 0.3927)"
        ])
    ]
    draw_card(68.5, 6, 29, 84, c_eval, c_eval_border, "STAGE 4: RESEARCH", "Empirical Research Validation", s4_sections)

    # Directed Flow Arrows
    arrow_props = dict(facecolor="#1c2833", edgecolor="#1c2833", width=2.0, headwidth=7.5, shrink=0.04)

    # Stage 1 -> Stage 2
    ax.annotate("", xy=(34.5, 71.5), xytext=(31.5, 71.5), arrowprops=arrow_props)
    # Stage 2 -> Stage 3
    ax.annotate("", xy=(50.0, 50.0), xytext=(50.0, 53.0), arrowprops=arrow_props)
    # Stage 1 -> Stage 3
    ax.annotate("", xy=(34.5, 28.0), xytext=(31.5, 28.0), arrowprops=arrow_props)
    # Stage 3 -> Stage 4
    ax.annotate("", xy=(68.5, 28.0), xytext=(65.5, 28.0), arrowprops=arrow_props)
    # Stage 2 -> Stage 4
    ax.annotate("", xy=(68.5, 71.5), xytext=(65.5, 71.5), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    target_path = os.path.join(out_dir, "figure1_research_architecture_flowchart.png")
    generate_pipeline_flowchart(target_path)
    print("Updated figure 1 saved to:", target_path)
