"""
Explainability & SHAP Stability Module
Implements TreeSHAP explainers and temporal attribution stability analysis.
Addresses Theme A, Research Question 7 (RQ7).
Based on Zhang & Zhao (2026) and Giudici & Wu (2025).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import shap
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

class ESGExplainabilitySuite:
    """
    Computes SHAP feature attributions and evaluates cross-temporal explanation stability.
    """
    def __init__(self, model: Any, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def compute_shap_values(self, X: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Computes SHAP value matrix and mean absolute SHAP importance table.
        """
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list): # For classification models
            shap_matrix = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_matrix = shap_values

        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        return shap_matrix, importance_df

    @staticmethod
    def evaluate_temporal_stability(
        importance_window1: pd.DataFrame,
        importance_window2: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculates stability of SHAP explanations across two disjoint time windows:
        1. Spearman Rank Correlation (Rank stability of drivers)
        2. Cosine Similarity (Vector alignment of attribution magnitudes)
        """
        merged = pd.merge(
            importance_window1, importance_window2,
            on="feature", suffixes=("_win1", "_win2")
        )
        
        rho, p_val = spearmanr(merged["mean_abs_shap_win1"], merged["mean_abs_shap_win2"])
        
        v1 = merged["mean_abs_shap_win1"].values.reshape(1, -1)
        v2 = merged["mean_abs_shap_win2"].values.reshape(1, -1)
        cos_sim = float(cosine_similarity(v1, v2)[0, 0])

        return {
            "spearman_rank_stability": round(float(rho), 4),
            "rank_stability_pvalue": float(p_val),
            "cosine_similarity": round(cos_sim, 4)
        }
