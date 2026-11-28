"""
Cross-Dataset SHAP Attributions, Feature Harmonization & Pooled Domain Transfer Engine
Conducts comparative interpretability and extrapolates why models behave differently
between Dataset 1 (Global Multi-Agency Panel) and Dataset 2 (UK Green Finance Panel).
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, r2_score
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(name)s: %(message)s')

class CrossDatasetEmpiricalAnalyzer:
    """
    Performs SHAP attribution comparison, cross-market feature harmonization,
    and pooled generalization analysis across Global and UK panels.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def compute_dataset_shap(
        model: object,
        X_test: pd.DataFrame,
        feature_names: list
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Computes TreeSHAP attributions and mean absolute feature importance.
        """
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        if isinstance(shap_vals, list):
            shap_matrix = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            shap_matrix = shap_vals

        mean_abs = np.mean(np.abs(shap_matrix), axis=0)
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        return shap_matrix, imp_df

    @staticmethod
    def harmonize_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Harmonizes heterogeneous market variables into unified dimensionless constructs:
        1. harm_scale: log(assets) standardized
        2. harm_leverage: debt ratio standardized
        3. harm_financial_health: composite Altman Z (D1) / Interest Coverage (D2)
        4. harm_esg_score: consensus ESG (D1) / SECR carbon score (D2)
        5. harm_discrepancy: greenwashing gap (D1) / RDI gap (D2)
        6. distress_t1: binary distress indicator in t+1
        """
        scaler = StandardScaler()

        # Dataset 1 Harmonization (Global)
        h1 = pd.DataFrame()
        h1["dataset_origin"] = ["Global_D1"] * len(df1)
        h1["firm_id"] = df1["firm_id"]
        h1["year"] = df1["year"]
        h1["harm_scale"] = scaler.fit_transform(df1[["log_assets"]])
        h1["harm_leverage"] = scaler.fit_transform(df1[["leverage"]])
        h1["harm_financial_health"] = scaler.fit_transform(df1[["altman_z"]])
        h1["harm_esg_score"] = scaler.fit_transform(df1[["consensus_esg"]])
        h1["harm_discrepancy"] = scaler.fit_transform(df1[["greenwashing_gap"]])
        h1["distress_t1"] = df1["financial_distress_t1"]

        # Dataset 2 Harmonization (UK Green Finance)
        h2 = pd.DataFrame()
        h2["dataset_origin"] = ["UK_D2"] * len(df2)
        h2["firm_id"] = df2["firm_id"]
        h2["year"] = df2["year"]
        h2["harm_scale"] = scaler.fit_transform(df2[["log_assets"]])
        h2["harm_leverage"] = scaler.fit_transform(df2[["leverage"]])
        h2["harm_financial_health"] = scaler.fit_transform(df2[["interest_coverage"]])
        h2["harm_esg_score"] = scaler.fit_transform(df2[["secr_carbon_performance"]])
        h2["harm_discrepancy"] = scaler.fit_transform(df2[["rdi_gap"]])
        h2["distress_t1"] = df2["credit_distress_t1"]

        # Combined Pooled Dataset
        h_combined = pd.concat([h1, h2], ignore_index=True)

        return h1, h2, h_combined

def plot_figure13_comparative_shap(
    imp_d1: pd.DataFrame,
    imp_d2: pd.DataFrame,
    output_path: str
):
    """
    Figure 13: Side-by-Side TreeSHAP Attributions across Global vs UK Panels
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Dataset 1 (Global Panel)
    ax1 = axes[0]
    sns.barplot(data=imp_d1.head(8), x="mean_abs_shap", y="feature", palette="Blues_r", ax=ax1)
    ax1.set_title("Panel A: Dataset 1 (Global Panel) - TreeSHAP Attributions", fontweight="bold")
    ax1.set_xlabel("Mean |SHAP Value| (Predictive Impact)")
    ax1.set_ylabel("Feature")
    for p in ax1.patches:
        w = p.get_width()
        ax1.annotate(f"{w:.3f}", (w, p.get_y() + p.get_height() / 2.),
                     ha="left", va="center", xytext=(5, 0), textcoords="offset points", fontsize=9.5)

    # Panel B: Dataset 2 (UK Green Finance Panel)
    ax2 = axes[1]
    sns.barplot(data=imp_d2.head(8), x="mean_abs_shap", y="feature", palette="Greens_r", ax=ax2)
    ax2.set_title("Panel B: Dataset 2 (UK Green Finance) - TreeSHAP Attributions", fontweight="bold")
    ax2.set_xlabel("Mean |SHAP Value| (Predictive Impact)")
    ax2.set_ylabel("Feature")
    for p in ax2.patches:
        w = p.get_width()
        ax2.annotate(f"{w:.3f}", (w, p.get_y() + p.get_height() / 2.),
                     ha="left", va="center", xytext=(5, 0), textcoords="offset points", fontsize=9.5)

    plt.suptitle("Figure 13: TreeSHAP Feature Attribution Divergence Between Global and UK Regulatory Regimes", y=1.03, fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_figure14_domain_transfer(
    transfer_matrix: pd.DataFrame,
    output_path: str
):
    """
    Figure 14: Cross-Dataset Harmonization, Pooled Generalization & Domain Transfer Matrix
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    sns.heatmap(
        transfer_matrix, annot=True, fmt=".4f", cmap="YlGnBu",
        cbar_kws={"label": "Out-of-Time ROC-AUC"}, linewidths=1.5, ax=ax
    )
    ax.set_title("Figure 14: Cross-Market Harmonization & Domain Transfer Matrix (Out-of-Time ROC-AUC)", fontweight="bold", pad=15)
    ax.set_xlabel("Test Environment (Target Domain)", fontweight="bold")
    ax.set_ylabel("Training Environment (Source Domain)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d1_path = os.path.join(base_dir, "data", "processed", "esg_financial_panel.parquet")
    d2_path = os.path.join(base_dir, "data", "processed", "uk_tcfd_green_finance_panel.parquet")
    metrics_dir = os.path.join(base_dir, "outputs", "metrics")
    figures_dir = os.path.join(base_dir, "outputs", "figures")

    df1 = pd.read_parquet(d1_path)
    df2 = pd.read_parquet(d2_path)

    analyzer = CrossDatasetEmpiricalAnalyzer(random_state=42)

    # 1. Dataset 1 SHAP Computation
    feats_d1 = [
        "log_assets", "leverage", "roa", "current_ratio", 
        "operating_margin", "debt_to_equity", "tobins_q", "volatility", "altman_z",
        "pillar_e", "pillar_s", "pillar_g", "greenwashing_gap"
    ]
    train_d1 = df1[df1["year"] <= 2022]
    test_d1 = df1[df1["year"] > 2022]
    
    m_d1 = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=1)
    m_d1.fit(train_d1[feats_d1], train_d1["financial_distress_t1"])
    _, imp_d1 = analyzer.compute_dataset_shap(m_d1, test_d1[feats_d1], feats_d1)

    # 2. Dataset 2 SHAP Computation
    feats_d2 = [
        "log_assets", "leverage", "interest_coverage", "ebitda_margin",
        "is_green_bond_issuer", "green_capex_ratio", "secr_carbon_performance",
        "tcfd_governance_score", "nlp_sentiment", "fls_vagueness", "rdi_gap"
    ]
    train_d2 = df2[df2["year"] <= 2022]
    test_d2 = df2[df2["year"] > 2022]

    m_d2 = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=1)
    m_d2.fit(train_d2[feats_d2], train_d2["credit_distress_t1"])
    _, imp_d2 = analyzer.compute_dataset_shap(m_d2, test_d2[feats_d2], feats_d2)

    # Plot Figure 13
    fig13_path = os.path.join(figures_dir, "figure13_comparative_shap_attributions.png")
    plot_figure13_comparative_shap(imp_d1, imp_d2, fig13_path)
    logger.info(f"Generated Figure 13: {fig13_path}")

    # 3. Cross-Market Harmonization & Pooling
    logger.info("\nHarmonizing variables and building pooled dataset (17,600 observations)...")
    h1, h2, h_combined = analyzer.harmonize_datasets(df1, df2)

    harm_parquet_path = os.path.join(base_dir, "data", "processed", "harmonized_pooled_panel.parquet")
    h_combined.to_parquet(harm_parquet_path, index=False)
    logger.info(f"Saved harmonized pooled panel ({len(h_combined)} rows) to {harm_parquet_path}")

    # 4. Domain Transfer Evaluation (Train on D1, Test on D2, Train on Combined, etc.)
    harm_feats = ["harm_scale", "harm_leverage", "harm_financial_health", "harm_esg_score", "harm_discrepancy"]

    train_h1 = h1[h1["year"] <= 2022]
    test_h1 = h1[h1["year"] > 2022]

    train_h2 = h2[h2["year"] <= 2022]
    test_h2 = h2[h2["year"] > 2022]

    train_comb = h_combined[h_combined["year"] <= 2022]
    test_comb = h_combined[h_combined["year"] > 2022]

    # Model training across domains
    m_h1 = lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1)
    m_h1.fit(train_h1[harm_feats], train_h1["distress_t1"])

    m_h2 = lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1)
    m_h2.fit(train_h2[harm_feats], train_h2["distress_t1"])

    m_comb = lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1)
    m_comb.fit(train_comb[harm_feats], train_comb["distress_t1"])

    # Predictions matrix
    def get_auc(m, X, y):
        return round(float(roc_auc_score(y, m.predict_proba(X)[:, 1])), 4)

    transfer_results = {
        "Train D1 (Global)": {
            "Test D1 (Global)": get_auc(m_h1, test_h1[harm_feats], test_h1["distress_t1"]),
            "Test D2 (UK)": get_auc(m_h1, test_h2[harm_feats], test_h2["distress_t1"]),
            "Test Pooled": get_auc(m_h1, test_comb[harm_feats], test_comb["distress_t1"])
        },
        "Train D2 (UK)": {
            "Test D1 (Global)": get_auc(m_h2, test_h1[harm_feats], test_h1["distress_t1"]),
            "Test D2 (UK)": get_auc(m_h2, test_h2[harm_feats], test_h2["distress_t1"]),
            "Test Pooled": get_auc(m_h2, test_comb[harm_feats], test_comb["distress_t1"])
        },
        "Train Pooled (D1+D2)": {
            "Test D1 (Global)": get_auc(m_comb, test_h1[harm_feats], test_h1["distress_t1"]),
            "Test D2 (UK)": get_auc(m_comb, test_h2[harm_feats], test_h2["distress_t1"]),
            "Test Pooled": get_auc(m_comb, test_comb[harm_feats], test_comb["distress_t1"])
        }
    }

    transfer_df = pd.DataFrame(transfer_results).T
    logger.info("\n--- CROSS-DOMAIN TRANSFER ROC-AUC MATRIX ---")
    logger.info(transfer_df)

    # Plot Figure 14
    fig14_path = os.path.join(figures_dir, "figure14_harmonized_pooled_domain_transfer.png")
    plot_figure14_domain_transfer(transfer_df, fig14_path)
    logger.info(f"Generated Figure 14: {fig14_path}")

    # Save metrics
    metrics_path = os.path.join(metrics_dir, "cross_dataset_harmonized_results.json")
    with open(metrics_path, "w") as f:
        json.dump(transfer_results, f, indent=4)
    logger.info(f"Saved cross-dataset metrics to {metrics_path}")
