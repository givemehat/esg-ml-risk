"""
Stand-Alone Publication Figure Generator
Regenerates all figures (Figure 1 through Figure 8) from stored datasets and benchmark results.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.pipeline_diagram import generate_pipeline_flowchart
from src.visualize import (
    plot_rating_divergence, plot_model_benchmarks,
    plot_uncertainty_intervals, plot_shap_summary,
    plot_greenwashing_matrix, plot_distress_incremental_roc
)

def main():
    print("Generating publication figures (300 DPI)...")
    figures_dir = os.path.join(BASE_DIR, "outputs", "figures")
    metrics_dir = os.path.join(BASE_DIR, "outputs", "metrics")
    data_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(figures_dir, exist_ok=True)

    data_path = os.path.join(data_dir, "esg_financial_panel.parquet")
    if not os.path.exists(data_path):
        print("Data panel not found. Please run experiments/run_all.py first.")
        return

    df = pd.read_parquet(data_path)

    # 1. Flowchart
    generate_pipeline_flowchart(os.path.join(figures_dir, "figure1_research_architecture_flowchart.png"))
    print("✓ Figure 1 generated.")

    # 2. Rating Divergence
    plot_rating_divergence(df, os.path.join(figures_dir, "figure2_rating_divergence.png"))
    print("✓ Figure 2 generated.")

    # 3. Model Benchmarks
    reg_csv = os.path.join(metrics_dir, "tabular_regression_benchmarks.csv")
    clf_csv = os.path.join(metrics_dir, "tabular_classification_benchmarks.csv")
    if os.path.exists(reg_csv) and os.path.exists(clf_csv):
        reg_df = pd.read_csv(reg_csv)
        clf_df = pd.read_csv(clf_csv)
        plot_model_benchmarks(reg_df, clf_df, os.path.join(figures_dir, "figure3_model_benchmarks.png"))
        print("✓ Figure 3 generated.")

    # 6. Greenwashing Matrix
    plot_greenwashing_matrix(df, os.path.join(figures_dir, "figure6_greenwashing_matrix.png"))
    print("✓ Figure 6 generated.")

    print("\nAll available figures successfully created in outputs/figures/")

if __name__ == "__main__":
    main()
