"""
Uncertainty Quantification Module
Implements Quantile Gradient Boosting & Conformal Prediction for ESG score prediction intervals.
Addresses Theme A, Research Question 1 (RQ1).
Based on Caprioli et al. (2024) and Zhang & Zhao (2026).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor

class ESGUncertaintyEstimator:
    """
    Quantifies predictive uncertainty for corporate ESG scores via:
    1. Quantile Regression (p10, p50, p90)
    2. Split Conformal Prediction (Finite-sample distribution-free coverage guarantee)
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.q_low_model = None
        self.q_med_model = None
        self.q_high_model = None
        self.conformal_q = None

    def fit_quantile_regressors(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        alpha_low: float = 0.10,
        alpha_high: float = 0.90
    ):
        """
        Fits pinball loss (quantile objective) models for lower, median, and upper bounds.
        """
        self.q_low_model = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha_low,
            n_estimators=150, max_depth=5, learning_rate=0.05,
            random_state=self.random_state, verbose=-1
        )
        self.q_med_model = lgb.LGBMRegressor(
            objective="quantile", alpha=0.50,
            n_estimators=150, max_depth=5, learning_rate=0.05,
            random_state=self.random_state, verbose=-1
        )
        self.q_high_model = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha_high,
            n_estimators=150, max_depth=5, learning_rate=0.05,
            random_state=self.random_state, verbose=-1
        )

        self.q_low_model.fit(X_train, y_train)
        self.q_med_model.fit(X_train, y_train)
        self.q_high_model.fit(X_train, y_train)

    def predict_quantile_intervals(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates lower, median, and upper quantile predictions.
        """
        pred_low = self.q_low_model.predict(X_test)
        pred_med = self.q_med_model.predict(X_test)
        pred_high = self.q_high_model.predict(X_test)
        
        # Enforce monotonicity: low <= med <= high
        pred_low = np.minimum(pred_low, pred_med)
        pred_high = np.maximum(pred_high, pred_med)
        return pred_low, pred_med, pred_high

    def calibrate_conformal_prediction(
        self,
        base_model: Any,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
        confidence_level: float = 0.90
    ):
        """
        Calibrates residual nonconformity score on an unseen calibration set (e.g. 2022 data).
        q_val satisfies: P(y_test in [y_hat - q_val, y_hat + q_val]) >= 1 - alpha
        """
        preds_calib = base_model.predict(X_calib)
        residuals = np.abs(y_calib.values - preds_calib)
        
        n = len(residuals)
        alpha = 1.0 - confidence_level
        k = int(np.ceil((1.0 - alpha) * (n + 1)))
        k = min(max(k, 1), n)
        
        self.conformal_q = np.sort(residuals)[k - 1]
        return self.conformal_q

    def predict_conformal_intervals(
        self,
        base_model: Any,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies conformal margin to point predictions.
        """
        point_preds = base_model.predict(X_test)
        lower_bound = np.clip(point_preds - self.conformal_q, 0, 100)
        upper_bound = np.clip(point_preds + self.conformal_q, 0, 100)
        return lower_bound, point_preds, upper_bound

    @staticmethod
    def evaluate_intervals(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        alpha: float = 0.20
    ) -> Dict[str, float]:
        """
        Computes PICP (Coverage), MPIW (Sharpness / Width), and Winkler Score.
        Winkler score penalizes intervals that miss the target:
        Score = Width + (2/alpha)*(lower - y) if y < lower + (2/alpha)*(y - upper) if y > upper
        """
        covered = (y_true >= lower) & (y_true <= upper)
        picp = float(np.mean(covered))
        
        width = upper - lower
        mpiw = float(np.mean(width))
        
        # Winkler Score
        winkler_penalties = width.copy()
        miss_below = y_true < lower
        miss_above = y_true > upper
        winkler_penalties[miss_below] += (2.0 / alpha) * (lower[miss_below] - y_true[miss_below])
        winkler_penalties[miss_above] += (2.0 / alpha) * (y_true[miss_above] - upper[miss_above])
        winkler_score = float(np.mean(winkler_penalties))

        return {
            "PICP (Coverage)": round(picp, 4),
            "MPIW (Mean Width)": round(mpiw, 4),
            "Winkler Score": round(winkler_score, 4)
        }
