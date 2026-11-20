"""
Research Pipeline Architecture & Flowchart Generator
Generates Figure 1: Comprehensive ML & NLP Research Architecture for ESG & Financial Risk.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_pipeline_flowchart(output_path: str):
    """
    Draws a publication-quality end-to-end research methodology diagram:
    Stage 1: Multi-Agency Data & Text Ingestion
    Stage 2: Strict Out-of-Time Temporal Splitting
    Stage 3: Multi-Model Machine Learning Core (Tabular + Transformers + Quantiles)
    Stage 4: Research Questions & Empirical Validation (Themes A, B, C)
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Color Palette
    c_data = "#e8f4f8"
    c_data_border = "#2980b9"
    c_split = "#fef9e7"
    c_split_border = "#f39c12"
    c_model = "#eafaf1"
    c_model_border = "#27ae60"
    c_eval = "#fcedeb"
    c_eval_border = "#c0392b"

    def draw_box(x, y, w, h, bg, border, title, items, badge=""):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.8,rounding_size=1.5",
            facecolor=bg, edgecolor=border, linewidth=2, zorder=2
        )
        ax.add_patch(rect)
        if badge:
            badge_rect = patches.FancyBboxPatch(
                (x + 2, y + h - 4), 22, 4.5,
                boxstyle="round,pad=0.3,rounding_size=0.8",
                facecolor=border, edgecolor=border, zorder=3
            )
            ax.add_patch(badge_rect)
            ax.text(x + 13, y + h - 2, badge, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", zorder=4)

        ax.text(x + w/2, y + h - 8, title, ha="center", va="center", color="#2c3e50", fontsize=11, fontweight="bold", zorder=3)
        
        start_y = y + h - 14
        for item in items:
            ax.text(x + 4, start_y, item, ha="left", va="center", color="#34495e", fontsize=9.2, zorder=3)
            start_y -= 4.2

    # Title
    ax.text(50, 96, "End-to-End Machine Learning Framework for ESG Assessment & Financial Risk", 
            ha="center", va="center", fontsize=16, fontweight="bold", color="#1a252f")
    ax.text(50, 92.5, "Rajnish Singh · Medhavie Gupta · Katyayani Jaiswal (Gautam Buddha Univ.) | Guide: Prof. Valentina Lagasio (Sapienza)",
            ha="center", va="center", fontsize=11, fontstyle="italic", color="#566573")

    # Stage 1: Data Ingestion (Left Column)
    draw_box(
        3, 48, 28, 38, c_data, c_data_border,
        "Stage 1: Multi-Source Data Panel",
        [
            "• Orbis Financials (Assets, ROA, Z-score)",
            "• Leverage, Tobin's Q, Volatility",
            "• Multi-Agency ESG (Refinitiv, MSCI, Sust.)",
            "• Pillar Breakdown: E, S, and G scores",
            "• Corporate Disclosures (10-K & CSR text)",
            "• 12,000 Firm-Years across 1,500 Global Firms"
        ],
        badge="STAGE 1: DATA"
    )

    # Stage 2: Temporal Split (Top Middle)
    draw_box(
        36, 68, 27, 18, c_split, c_split_border,
        "Stage 2: Strict Temporal Split",
        [
            "• Train Window: 2018–2022 (7,500 obs)",
            "• Calibration Split: 2022 for Conformal",
            "• Strict Out-of-Time Test: 2023–2025",
            "• Zero Future Information Leakage"
        ],
        badge="STAGE 2: SPLIT"
    )

    # Stage 3: Core Modeling Engine (Bottom Middle)
    draw_box(
        36, 12, 27, 50, c_model, c_model_border,
        "Stage 3: Modeling Engine",
        [
            "Tabular ML Suite:",
            "  - OLS/Ridge Baselines",
            "  - Random Forest & Extra Trees",
            "  - XGBoost & LightGBM",
            "",
            "NLP & Transformer Engine:",
            "  - Contextual DistilBERT / FinBERT",
            "  - Polarity & Vagueness Density",
            "  - Greenwashing Gap: Rhetoric vs E-score",
            "",
            "Uncertainty Engine:",
            "  - Quantile Gradient Boosting (p10, p90)",
            "  - Inductive Conformal Prediction (90%)"
        ],
        badge="STAGE 3: MODELS"
    )

    # Stage 4: Research Themes & Questions (Right Column)
    draw_box(
        68, 12, 29, 74, c_eval, c_eval_border,
        "Stage 4: Research Validation",
        [
            "Theme A: Scores & Uncertainty",
            "  • RQ1: Conformal & Quantile Intervals",
            "    (PICP Coverage = 86.7%, Winkler Score)",
            "  • RQ7: TreeSHAP Feature Stability",
            "    (Spearman rho = 0.66, Cosine Sim = 0.99)",
            "  • RQ8: Selective Coverage & Missingness",
            "",
            "Theme B: Distress & Risk (Out-of-Time)",
            "  • RQ2: Disagreement Incremental Signal",
            "  • RQ5: Greenwashing Risk Distress Penalty",
            "  • RQ6: Pillar Importance (E vs S vs G)",
            "    (M1: Financials vs M2-M5 Incremental)",
            "",
            "Theme C: Text & Greenwashing",
            "  • RQ3: Rhetoric vs Empirical Metrics",
            "  • RQ4: Rating Disagreement Correlation",
            "  • Transformer vs TF-IDF Detection"
        ],
        badge="STAGE 4: RESEARCH QUESTIONS"
    )

    # Directed Arrows
    arrow_props = dict(facecolor="#2c3e50", edgecolor="#2c3e50", width=1.5, headwidth=7, shrink=0.05)
    
    # S1 -> S2
    ax.annotate("", xy=(36, 77), xytext=(31, 77), arrowprops=arrow_props)
    # S2 -> S3
    ax.annotate("", xy=(49.5, 62), xytext=(49.5, 68), arrowprops=arrow_props)
    # S1 -> S3
    ax.annotate("", xy=(36, 37), xytext=(31, 52), arrowprops=arrow_props)
    # S3 -> S4
    ax.annotate("", xy=(68, 45), xytext=(63, 45), arrowprops=arrow_props)
    # S2 -> S4
    ax.annotate("", xy=(68, 75), xytext=(63, 75), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "figure1_research_architecture_flowchart.png")
    generate_pipeline_flowchart(out_file)
    print(f"Generated Figure 1 Flowchart: {out_file}")
