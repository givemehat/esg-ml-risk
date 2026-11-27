"""
Tabular Machine Learning Benchmarks
Trains and benchmarks Ridge/Logistic, Random Forest, Extra Trees, XGBoost, and LightGBM.
Based on Zhang & Zhao (2026), Dossa et al. (2025), and Raza et al. (2026).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, brier_score_loss, f1_score
)
import xgboost as xgb
import lightgbm as lgb

FEATURE_COLS_FINANCIAL = [
    "log_assets", "leverage", "roa", "current_ratio", 
    "operating_margin", "debt_to_equity", "tobins_q", "volatility", "altman_z"
]

FEATURE_COLS_ESG = [
    "pillar_e", "pillar_s", "pillar_g", "consensus_esg", "esg_disagreement"
]

class TabularESGModelSuite:
    """
    Unified training, tuning, and evaluation suite for ESG score prediction
    and out-of-time financial distress modeling.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models_reg: Dict[str, Any] = {}
        self.models_clf: Dict[str, Any] = {}
        
    def _init_regressors(self) -> Dict[str, Any]:
        return {
            "Ridge Baseline": Ridge(alpha=1.0, random_state=self.random_state),
            "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=10, random_state=self.random_state, n_jobs=1),
            "Extra Trees": ExtraTreesRegressor(n_estimators=150, max_depth=10, random_state=self.random_state, n_jobs=1),
            "XGBoost": xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=self.random_state, n_jobs=1),
            "LightGBM": lgb.LGBMRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=self.random_state, n_jobs=1, verbose=-1)
        }

    def _init_classifiers(self) -> Dict[str, Any]:
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=self.random_state, n_jobs=1),
            "Extra Trees": ExtraTreesClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=self.random_state, n_jobs=1),
            "XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, scale_pos_weight=3.0, random_state=self.random_state, n_jobs=1),
            "LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, class_weight="balanced", random_state=self.random_state, n_jobs=1, verbose=-1)
        }

    def train_evaluate_regression(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: list,
        target_col: str = "consensus_esg"
    ) -> pd.DataFrame:
        """
        Trains regression models on historical data and tests on future out-of-time data.
        Returns comparative metrics (RMSE, MAE, R2).
        """
        X_train, y_train = train_df[features], train_df[target_col]
        X_test, y_test = test_df[features], test_df[target_col]

        self.models_reg = self._init_regressors()
        results = []

        for name, model in self.models_reg.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            results.append({
                "Model": name,
                "Target": target_col,
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4)
            })

        return pd.DataFrame(results)

    def train_evaluate_classification(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: list,
        target_col: str = "financial_distress_t1"
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Trains classification models on historical data and evaluates on future out-of-time data.
        Returns comparative classification metrics and test probability predictions.
        """
        X_train, y_train = train_df[features], train_df[target_col]
        X_test, y_test = test_df[features], test_df[target_col]

        self.models_clf = self._init_classifiers()
        results = []
        pred_probs = {}

        for name, model in self.models_clf.items():
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = model.predict(X_test)
                
            pred_probs[name] = probs
            binary_preds = (probs >= 0.5).astype(int)

            roc_auc = roc_auc_score(y_test, probs)
            pr_auc = average_precision_score(y_test, probs)
            brier = brier_score_loss(y_test, probs)
            f1 = f1_score(y_test, binary_preds, zero_division=0)

            results.append({
                "Model": name,
                "Target": target_col,
                "ROC-AUC": round(roc_auc, 4),
                "PR-AUC": round(pr_auc, 4),
                "Brier Score": round(brier, 4),
                "F1-Score": round(f1, 4)
            })

        return pd.DataFrame(results), pred_probs
