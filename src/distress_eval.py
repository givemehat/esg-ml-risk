"""
Financial Distress & Incremental Information Evaluation Module
Evaluates incremental predictive power of ESG level, disagreement, pillars, and greenwashing
beyond traditional financial accounting ratios in strict out-of-time splits (2023-2025).
Addresses Theme B (RQ2, RQ5, RQ6).
Based on Raza et al. (2026), Liu et al. (2025), and Dossa et al. (2025).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import lightgbm as lgb
import xgboost as xgb

class FinancialDistressEvaluator:
    """
    Stepwise incremental test of ESG constructs for default/distress prediction.
    """
    FEATURE_SETS = {
        "M1: Financials Only": [
            "log_assets", "leverage", "roa", "current_ratio", 
            "operating_margin", "debt_to_equity", "tobins_q", "volatility", "altman_z"
        ],
        "M2: Financials + ESG Level": [
            "log_assets", "leverage", "roa", "current_ratio", 
            "operating_margin", "debt_to_equity", "tobins_q", "volatility", "altman_z",
            "consensus_esg"
        ],
        "M3: Financials + Individual Pillars (E, S, G)": [
            "log_assets", "leverage", "roa", "current_ratio", 
            "operating_margin", "debt_to_equity", "tobins_q", "volatility", "altman_z",
            "pillar_e", "pillar_s", "pillar_g"
        ],
        "M4: Financials + ESG + Disagreement (RQ2)": [
            "log_assets", "leverage", "roa", "current_ratio", 
            "operating_margin", "debt_to_equity", "tobins_q", "volatility", "altman_z",
            "consensus_esg", "esg_disagreement"
        ],
        "M5: Full Model (+ Greenwashing Gap) (RQ5)": [
            "log_assets", "leverage", "roa", "current_ratio", 
            "operating_margin", "debt_to_equity", "tobins_q", "volatility", "altman_z",
            "consensus_esg", "esg_disagreement", "greenwashing_gap", "nlp_vagueness"
        ]
    }

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run_incremental_comparison(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_type: str = "lightgbm",
        target_col: str = "financial_distress_t1"
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Trains models across all 5 feature sets and evaluates on strict out-of-time test set.
        """
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        results = []
        test_predictions = {}
        baseline_auc = None

        for spec_name, features in self.FEATURE_SETS.items():
            X_train = train_df[features]
            X_test = test_df[features]

            if model_type == "lightgbm":
                model = lgb.LGBMClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    class_weight="balanced", random_state=self.random_state,
                    verbose=-1
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    scale_pos_weight=3.0, random_state=self.random_state,
                    n_jobs=-1
                )

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            test_predictions[spec_name] = probs

            roc_auc = roc_auc_score(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)
            brier = brier_score_loss(y_test, probs)

            if baseline_auc is None:
                baseline_auc = roc_auc
                delta_auc = 0.0
            else:
                delta_auc = roc_auc - baseline_auc

            results.append({
                "Specification": spec_name,
                "Num Features": len(features),
                "ROC-AUC": round(roc_auc, 4),
                "Delta AUC vs M1": f"{delta_auc:+.4f}",
                "PR-AUC": round(pr_auc, 4),
                "Brier Score": round(brier, 4)
            })

        results_df = pd.DataFrame(results)
        return results_df, test_predictions
