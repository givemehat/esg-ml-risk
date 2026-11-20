"""
Visualization & Publication-Grade Chart Generator
Generates publication-quality figures (300 DPI) for all research questions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Publication style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 14

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

def plot_rating_divergence(df: pd.DataFrame, output_path: str):
    """
    Figure 2: Multi-Agency ESG Divergence (Berg et al. 2022 reproduction)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    
    agencies = [
        ("agency_refinitiv", "agency_msci", "Refinitiv vs. MSCI"),
        ("agency_refinitiv", "agency_sustainalytics", "Refinitiv vs. Sustainalytics"),
        ("agency_msci", "agency_sustainalytics", "MSCI vs. Sustainalytics")
    ]
    
    for idx, (x_col, y_col, title) in enumerate(agencies):
        ax = axes[idx]
        corr = df[[x_col, y_col]].corr().iloc[0, 1]
        
        sns.scatterplot(
            data=df.sample(min(1500, len(df)), random_state=42),
            x=x_col, y=y_col, alpha=0.35, s=20, color="#2b5c8f", ax=ax
        )
        
        # Identity line
        ax.plot([0, 100], [0, 100], color="#d62728", linestyle="--", linewidth=1.5, label="Perfect Agreement (r=1.0)")
        
        # Regression line
        sns.regplot(
            data=df, x=x_col, y=y_col, scatter=False,
            ax=ax, color="#1f77b4", line_kws={"linewidth": 2}
        )
        
        ax.set_title(f"{title}\n(Pearson r = {corr:.2f})", fontweight="bold")
        ax.set_xlabel(x_col.replace("agency_", "").title() + " Rating")
        if idx == 0:
            ax.set_ylabel(y_col.replace("agency_", "").title() + " Rating")
        else:
            ax.set_ylabel("")
        ax.set_xlim(10, 95)
        ax.set_ylim(10, 95)
        ax.legend(loc="upper left")

    plt.suptitle("Figure 2: Empirical Rating Divergence Across Three Major ESG Rating Agencies", y=1.03, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_model_benchmarks(reg_df: pd.DataFrame, clf_df: pd.DataFrame, output_path: str):
    """
    Figure 3: Tabular Benchmark Comparison (Regression RMSE & Classification ROC-AUC)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Regression RMSE
    ax1 = axes[0]
    sns.barplot(data=reg_df, x="Model", y="RMSE", palette="mako", ax=ax1)
    ax1.set_title("Panel A: ESG Score Prediction (RMSE - Lower is Better)", fontweight="bold")
    ax1.set_ylabel("Root Mean Squared Error")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=25)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="center", xytext=(0, 7), textcoords="offset points")

    # Subplot 2: Classification ROC-AUC
    ax2 = axes[1]
    sns.barplot(data=clf_df, x="Model", y="ROC-AUC", palette="viridis", ax=ax2)
    ax2.set_title("Panel B: Out-of-Time Distress Prediction (ROC-AUC - Higher is Better)", fontweight="bold")
    ax2.set_ylabel("Out-of-Time ROC-AUC (2023-2025)")
    ax2.set_xlabel("")
    ax2.set_ylim(0.5, 0.95)
    ax2.tick_params(axis="x", rotation=25)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha="center", va="center", xytext=(0, 7), textcoords="offset points")

    plt.suptitle("Figure 3: Machine Learning Tabular Benchmarks on Strict Out-of-Time Test Set", y=1.03, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_uncertainty_intervals(
    y_test: np.ndarray,
    preds_med: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    output_path: str
):
    """
    Figure 4: Prediction Intervals and Uncertainty Bands (Theme A, RQ1)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Panel A: Sample Firm Prediction Intervals
    ax1 = axes[0]
    sample_indices = np.argsort(preds_med[:60])
    sub_y = y_test[:60][sample_indices]
    sub_med = preds_med[:60][sample_indices]
    sub_low = lower[:60][sample_indices]
    sub_high = upper[:60][sample_indices]
    x_axis = np.arange(len(sub_y))

    ax1.fill_between(x_axis, sub_low, sub_high, color="#3498db", alpha=0.25, label="80% Prediction Interval")
    ax1.plot(x_axis, sub_med, color="#2980b9", linewidth=2, label="Median Prediction")
    ax1.scatter(x_axis, sub_y, color="#e74c3c", s=25, zorder=5, label="Observed Consensus ESG")
    ax1.set_title("Panel A: ESG Prediction Intervals (Sample 60 Test Firms)", fontweight="bold")
    ax1.set_xlabel("Firm Index (Sorted by Predicted Score)")
    ax1.set_ylabel("ESG Rating (0-100)")
    ax1.legend(loc="upper left")

    # Panel B: Interval Coverage Calibration
    ax2 = axes[1]
    widths = upper - lower
    covered = (y_test >= lower) & (y_test <= upper)
    picp = np.mean(covered)
    
    sns.histplot(widths, kde=True, color="#27ae60", ax=ax2, bins=25)
    ax2.axvline(np.mean(widths), color="#c0392b", linestyle="--", linewidth=2, 
                label=f"Mean Width (MPIW) = {np.mean(widths):.2f}")
    ax2.set_title(f"Panel B: Interval Width Distribution (Empirical PICP = {picp*100:.1f}%)", fontweight="bold")
    ax2.set_xlabel("Prediction Interval Width ($\hat{y}_{high} - \hat{y}_{low}$)")
    ax2.set_ylabel("Firm Count")
    ax2.legend(loc="upper right")

    plt.suptitle("Figure 4: Predictive Uncertainty & Conformal Interval Quantification (RQ1)", y=1.03, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_shap_summary(importance_df: pd.DataFrame, stability_df: pd.DataFrame, output_path: str):
    """
    Figure 5: TreeSHAP Feature Attributions and Cross-Temporal Stability (Theme A, RQ7)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Mean Absolute SHAP Importance
    ax1 = axes[0]
    top_feats = importance_df.head(10)
    sns.barplot(data=top_feats, y="feature", x="mean_abs_shap", palette="rocket", ax=ax1)
    ax1.set_title("Panel A: Global Feature Importance (TreeSHAP)", fontweight="bold")
    ax1.set_xlabel("Mean |SHAP Value| (Impact on Model Output)")
    ax1.set_ylabel("Feature")

    # Panel B: Temporal Attribution Stability
    ax2 = axes[1]
    sns.regplot(
        data=stability_df, x="mean_abs_shap_win1", y="mean_abs_shap_win2",
        ax=ax2, color="#2c3e50", scatter_kws={"s": 60, "alpha": 0.8}
    )
    for _, row in stability_df.iterrows():
        ax2.annotate(row["feature"], (row["mean_abs_shap_win1"] + 0.005, row["mean_abs_shap_win2"] + 0.005),
                     fontsize=9, alpha=0.85)

    ax2.set_title("Panel B: Temporal Stability (2018-2020 vs 2021-2022 Window)", fontweight="bold")
    ax2.set_xlabel("SHAP Importance (Historical Window 1)")
    ax2.set_ylabel("SHAP Importance (Historical Window 2)")

    plt.suptitle("Figure 5: Explainability & Temporal Feature Attribution Stability (RQ7)", y=1.03, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_greenwashing_matrix(df: pd.DataFrame, output_path: str):
    """
    Figure 6: Greenwashing Discrepancy Matrix (Theme C, RQ3 & RQ4)
    """
    plt.figure(figsize=(9, 7))
    sample_df = df.sample(min(2000, len(df)), random_state=42)
    
    # Scatter plot of Text Sentiment vs Real Environmental Performance
    scatter = plt.scatter(
        sample_df["pillar_e"], sample_df["nlp_sentiment"] * 100,
        c=sample_df["greenwashing_gap"], cmap="coolwarm", s=30, alpha=0.6, edgecolors="none"
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Greenwashing Discrepancy Gap ($Rhetoric - Performance$)", rotation=270, labelpad=18)

    # Quadrant lines at medians
    median_e = sample_df["pillar_e"].median()
    median_sent = (sample_df["nlp_sentiment"] * 100).median()
    plt.axvline(median_e, color="gray", linestyle="--", alpha=0.6)
    plt.axhline(median_sent, color="gray", linestyle="--", alpha=0.6)

    # Annotate quadrants
    plt.text(18, 92, "High Greenwash Risk\n(Low Real E, High Rosy Text)", 
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec="#c62828", alpha=0.8), fontweight="bold")
    plt.text(68, 42, "Authentic Green Leaders\n(High Real E, Measured Text)", 
             bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#2e7d32", alpha=0.8), fontweight="bold")

    plt.title("Figure 6: Greenwashing Risk Matrix (Disclosure Sentiment vs Real Environmental Performance)", fontweight="bold", pad=15)
    plt.xlabel("Quantitative Environmental Pillar Score (Real Performance)")
    plt.ylabel("NLP Disclosure Sentiment Score (Corporate Tone)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_distress_incremental_roc(
    y_test: np.ndarray,
    spec_preds: dict,
    output_path: str
):
    """
    Figure 7: Out-of-Time Financial Distress ROC & PR Curves (Theme B, RQ2 & RQ5)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    ax_roc = axes[0]
    ax_pr = axes[1]
    
    palette = ["#7f8c8d", "#2980b9", "#8e44ad", "#e67e22", "#c0392b"]

    for idx, (spec_name, probs) in enumerate(spec_preds.items()):
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{spec_name} (AUC={roc_val:.3f})", color=palette[idx % len(palette)], linewidth=2)

        # PR curve
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_val = auc(rec, prec)
        ax_pr.plot(rec, prec, label=f"{spec_name} (PR={pr_val:.3f})", color=palette[idx % len(palette)], linewidth=2)

    # Reference lines
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.50)")
    ax_roc.set_title("Panel A: Out-of-Time Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")

    base_rate = np.mean(y_test)
    ax_pr.axhline(base_rate, color="black", linestyle="--", alpha=0.5, label=f"Base Rate ({base_rate:.2f})")
    ax_pr.set_title("Panel B: Precision-Recall Curve (Out-of-Time 2023-2025)", fontweight="bold")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="upper right")

    plt.suptitle("Figure 7: Incremental Information of ESG Disagreement & Greenwashing on Distress Risk", y=1.03, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
