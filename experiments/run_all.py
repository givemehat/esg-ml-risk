"""
Master Experiment Runner
Executes the full research pipeline across Themes A, B, and C.
Evaluates RQ1, RQ2, RQ3, RQ4, RQ5, RQ6, RQ7, RQ8.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.data_loader import generate_esg_financial_dataset, get_temporal_splits
from src.nlp_greenwashing import ESGGreenwashingAnalyzer
from src.models_tabular import (
    TabularESGModelSuite, FEATURE_COLS_FINANCIAL, FEATURE_COLS_ESG
)
from src.uncertainty import ESGUncertaintyEstimator
from src.explainability import ESGExplainabilitySuite
from src.distress_eval import FinancialDistressEvaluator
from src.pipeline_diagram import generate_pipeline_flowchart
from src.transformer_greenwash import compare_transformer_vs_tfidf, plot_transformer_comparison
from src.visualize import (
    plot_rating_divergence, plot_model_benchmarks,
    plot_uncertainty_intervals, plot_shap_summary,
    plot_greenwashing_matrix, plot_distress_incremental_roc
)

def main():
    print("=" * 80)
    print("STARTING ESG MACHINE LEARNING & FINANCIAL RISK RESEARCH PIPELINE")
    print("Sapienza University of Rome & Gautam Buddha University")
    print("Meeting Date: 20 November 2026 | Guide: Prof. Valentina Lagasio")
    print("=" * 80)

    # 1. Paths & Directories
    data_dir = os.path.join(BASE_DIR, "data", "processed")
    metrics_dir = os.path.join(BASE_DIR, "outputs", "metrics")
    figures_dir = os.path.join(BASE_DIR, "outputs", "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Flowchart
    print("\n[Step 1/8] Generating Research Architecture Flowchart (Figure 1)...")
    generate_pipeline_flowchart(os.path.join(figures_dir, "figure1_research_architecture_flowchart.png"))

    # 2. Paths & Directories
    data_path = os.path.join(data_dir, "esg_financial_panel.parquet")
    if not os.path.exists(data_path):
        print("\n[Step 2/8] Generating synthetic multi-agency ESG panel dataset...")
        df = generate_esg_financial_dataset(n_firms=1500, seed=42)
        df.to_parquet(data_path, index=False)
    else:
        print(f"\n[Step 2/8] Loading existing panel dataset from {data_path}...")
        df = pd.read_parquet(data_path)

    print(f"Total Observations: {len(df)} across {df['firm_id'].nunique()} firms and {df['year'].nunique()} years (2018-2025).")

    # 2. NLP & Greenwashing Analysis
    print("\n[Step 3/8] Running NLP Greenwashing & Linguistic Analysis (Theme C)...")
    nlp_analyzer = ESGGreenwashingAnalyzer()
    text_features = nlp_analyzer.extract_text_features(df["csr_disclosure"].tolist())
    df["vague_density"] = text_features["vague_density"]
    df["concrete_density"] = text_features["concrete_density"]
    df["vagueness_ratio"] = text_features["vagueness_ratio"]

    # RQ4 Test: Greenwashing Gap vs Rating Disagreement
    rq4_correlations = nlp_analyzer.test_greenwashing_disagreement_correlation(df)
    with open(os.path.join(metrics_dir, "rq4_greenwash_disagreement_corr.json"), "w") as f:
        json.dump(rq4_correlations, f, indent=4)
    print(f"  -> RQ4 Disagreement Correlation: Pearson r = {rq4_correlations['pearson_corr']:.4f} (p={rq4_correlations['pearson_pvalue']:.4e})")

    # Run Transformer Contextual Benchmark
    print("  -> Benchmarking Transformer Self-Attention vs TF-IDF for Greenwash Detection (Figure 8)...")
    gw_labels = (df["greenwashing_gap"] > 0.20).astype(int).values
    train_idx = np.where(df["year"] <= 2022)[0]
    test_idx = np.where(df["year"] > 2022)[0]
    m_tfidf, m_tf, embeddings = compare_transformer_vs_tfidf(df["csr_disclosure"].tolist(), gw_labels, train_idx, test_idx)
    with open(os.path.join(metrics_dir, "transformer_vs_tfidf_nlp.json"), "w") as f:
        json.dump({"TF_IDF": m_tfidf, "Transformer": m_tf}, f, indent=4)
    plot_transformer_comparison(m_tfidf, m_tf, embeddings, gw_labels, os.path.join(figures_dir, "figure8_transformer_nlp_greenwash.png"))

    # Plot Figure 2 (Rating Divergence) & Figure 6 (Greenwashing Matrix)
    print("  -> Generating Figure 2 (Rating Divergence) and Figure 6 (Greenwashing Matrix)...")
    plot_rating_divergence(df, os.path.join(figures_dir, "figure2_rating_divergence.png"))
    plot_greenwashing_matrix(df, os.path.join(figures_dir, "figure6_greenwashing_matrix.png"))

    # 3. Strict Out-of-Time Temporal Split (Raza et al. 2026)
    print("\n[Step 4/8] Creating strict out-of-time temporal split (Train: 2018-2022, Test: 2023-2025)...")
    train_df, test_df = get_temporal_splits(df, train_end_year=2022)
    print(f"  -> Training sample: {len(train_df)} firm-years (2018-2022)")
    print(f"  -> Out-of-time test sample: {len(test_df)} firm-years (2023-2025)")

    # 4. Tabular Benchmark Suite (Ridge, RF, Extra Trees, XGBoost, LightGBM)
    print("\n[Step 5/8] Training tabular ML models on ESG Score Prediction and Distress Classification...")
    tabular_suite = TabularESGModelSuite(random_state=42)
    
    # Task A: Consensus ESG Score Prediction
    all_features_reg = FEATURE_COLS_FINANCIAL + ["pillar_e", "pillar_s", "pillar_g"]
    reg_metrics = tabular_suite.train_evaluate_regression(
        train_df, test_df, features=all_features_reg, target_col="consensus_esg"
    )
    reg_metrics.to_csv(os.path.join(metrics_dir, "tabular_regression_benchmarks.csv"), index=False)
    print("\n  Tabular ESG Score Prediction Benchmark (Test: 2023-2025):")
    print(reg_metrics.to_string(index=False))

    # Task B: Out-of-Time Financial Distress Prediction
    clf_features = FEATURE_COLS_FINANCIAL + ["consensus_esg", "esg_disagreement"]
    clf_metrics, clf_pred_probs = tabular_suite.train_evaluate_classification(
        train_df, test_df, features=clf_features, target_col="financial_distress_t1"
    )
    clf_metrics.to_csv(os.path.join(metrics_dir, "tabular_classification_benchmarks.csv"), index=False)
    print("\n  Tabular Financial Distress Classification Benchmark (Test: 2023-2025):")
    print(clf_metrics.to_string(index=False))

    plot_model_benchmarks(reg_metrics, clf_metrics, os.path.join(figures_dir, "figure3_model_benchmarks.png"))

    # 5. Uncertainty Quantification & Conformal Prediction (Theme A, RQ1)
    print("\n[Step 6/8] Quantifying ESG Prediction Uncertainty (Quantile Regression & Conformal Intervals)...")
    uncertainty_suite = ESGUncertaintyEstimator(random_state=42)
    
    # Calibration set: year 2022; Core train: 2018-2021
    core_train = train_df[train_df["year"] <= 2021]
    calib_set = train_df[train_df["year"] == 2022]

    uncertainty_suite.fit_quantile_regressors(
        core_train[all_features_reg], core_train["consensus_esg"], alpha_low=0.10, alpha_high=0.90
    )
    q_low, q_med, q_high = uncertainty_suite.predict_quantile_intervals(test_df[all_features_reg])
    q_eval = uncertainty_suite.evaluate_intervals(test_df["consensus_esg"].values, q_low, q_high, alpha=0.20)

    # Conformal Prediction on LightGBM
    best_base_reg = tabular_suite.models_reg["LightGBM"]
    uncertainty_suite.calibrate_conformal_prediction(best_base_reg, calib_set[all_features_reg], calib_set["consensus_esg"], confidence_level=0.90)
    c_low, c_point, c_high = uncertainty_suite.predict_conformal_intervals(best_base_reg, test_df[all_features_reg])
    c_eval = uncertainty_suite.evaluate_intervals(test_df["consensus_esg"].values, c_low, c_high, alpha=0.10)

    uncertainty_results = {
        "Quantile_Regression_80pct": q_eval,
        "Conformal_Prediction_90pct": c_eval
    }
    with open(os.path.join(metrics_dir, "rq1_uncertainty_quantification.json"), "w") as f:
        json.dump(uncertainty_results, f, indent=4)
    print(f"  -> Quantile Interval (80% target): PICP = {q_eval['PICP (Coverage)']*100:.2f}%, MPIW = {q_eval['MPIW (Mean Width)']:.2f}")
    print(f"  -> Conformal Interval (90% target): PICP = {c_eval['PICP (Coverage)']*100:.2f}%, MPIW = {c_eval['MPIW (Mean Width)']:.2f}")

    plot_uncertainty_intervals(
        test_df["consensus_esg"].values, q_med, q_low, q_high,
        os.path.join(figures_dir, "figure4_uncertainty_intervals.png")
    )

    # 6. TreeSHAP & Temporal Attribution Stability (Theme A, RQ7)
    print("\n[Step 7/8] Computing TreeSHAP Values and Temporal Stability across Time Windows...")
    xgb_reg = tabular_suite.models_reg["XGBoost"]
    explainability_suite = ESGExplainabilitySuite(xgb_reg, all_features_reg)
    shap_matrix, global_importance = explainability_suite.compute_shap_values(test_df[all_features_reg])
    global_importance.to_csv(os.path.join(metrics_dir, "shap_global_feature_importance.csv"), index=False)

    # Cross-temporal window stability (Window 1: 2018-2020 vs Window 2: 2021-2022)
    win1_df = train_df[train_df["year"].isin([2018, 2019, 2020])]
    win2_df = train_df[train_df["year"].isin([2021, 2022])]
    
    m_win1 = tabular_suite.models_reg["XGBoost"]
    m_win1.fit(win1_df[all_features_reg], win1_df["consensus_esg"])
    _, imp_win1 = ESGExplainabilitySuite(m_win1, all_features_reg).compute_shap_values(win1_df[all_features_reg])

    m_win2 = tabular_suite.models_reg["XGBoost"]
    m_win2.fit(win2_df[all_features_reg], win2_df["consensus_esg"])
    _, imp_win2 = ESGExplainabilitySuite(m_win2, all_features_reg).compute_shap_values(win2_df[all_features_reg])

    stability_metrics = ESGExplainabilitySuite.evaluate_temporal_stability(imp_win1, imp_win2)
    with open(os.path.join(metrics_dir, "rq7_shap_temporal_stability.json"), "w") as f:
        json.dump(stability_metrics, f, indent=4)
    print(f"  -> SHAP Temporal Stability: Spearman Rank Corr = {stability_metrics['spearman_rank_stability']:.4f}, Cosine Sim = {stability_metrics['cosine_similarity']:.4f}")

    merged_stability_df = pd.merge(imp_win1, imp_win2, on="feature", suffixes=("_win1", "_win2"))
    plot_shap_summary(global_importance, merged_stability_df, os.path.join(figures_dir, "figure5_shap_stability.png"))

    # 7. Financial Distress Incremental Information (Theme B, RQ2, RQ5, RQ6)
    print("\n[Step 8/8] Evaluating Incremental Predictive Value on Financial Distress (M1 to M5)...")
    distress_evaluator = FinancialDistressEvaluator(random_state=42)
    incremental_results, distress_probs = distress_evaluator.run_incremental_comparison(
        train_df, test_df, model_type="lightgbm", target_col="financial_distress_t1"
    )
    incremental_results.to_csv(os.path.join(metrics_dir, "incremental_distress_specifications.csv"), index=False)
    print("\n  Incremental Specification Results on Out-of-Time Test Set (2023-2025):")
    print(incremental_results.to_string(index=False))

    plot_distress_incremental_roc(
        test_df["financial_distress_t1"].values, distress_probs,
        os.path.join(figures_dir, "figure7_distress_incremental_roc.png")
    )

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Metrics saved to: {metrics_dir}")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
