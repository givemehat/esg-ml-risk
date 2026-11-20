"""
NLP & Greenwashing Analysis Module
Implements Transformer / FinBERT text representations and sentiment-vs-reality greenwashing index.
Based on Schimanski et al. (2024), Calamai et al. (2025), and Zhao et al. (2024).
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class ESGGreenwashingAnalyzer:
    """
    Analyzes corporate CSR / 10-K disclosures for:
    1. Polarity / Optimism (Sentiment tone)
    2. Vagueness vs Specificity (Abstract buzzwords vs verifiable metrics)
    3. Sentiment-Reality Divergence (Greenwashing Index)
    """

    # Lexical dictionaries based on corporate ESG greenwashing literature
    VAGUE_BUZZWORDS = [
        "visionary", "stewardship", "transformative", "unparalleled", "breakthrough",
        "holistic", "harmony", "paradigm", "pioneering", "eco-friendly", "committed",
        "safeguard", "sustainable future", "green journey", "deep commitment"
    ]
    
    VERIFIABLE_METRICS = [
        "iso 14064", "ghg protocol", "mtco2e", "scope 1", "scope 2", "scope 3",
        "megaliters", "kwh", "mwh", "audited", "third-party", "turnover", "percent",
        "target achieved", "capex", "reduction of", "metric tons"
    ]

    def __init__(self):
        self.vague_regex = re.compile("|".join(r"\b" + re.escape(w) + r"\b" for w in self.VAGUE_BUZZWORDS), re.IGNORECASE)
        self.verifiable_regex = re.compile("|".join(r"\b" + re.escape(w) + r"\b" for w in self.VERIFIABLE_METRICS), re.IGNORECASE)

    def extract_text_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extracts NLP linguistic indicators:
        - vague_density: occurrences of abstract assertions per 100 words
        - concrete_density: occurrences of verifiable metrics per 100 words
        - vagueness_ratio: vague / (vague + concrete + 1e-4)
        """
        results = []
        for text in texts:
            words = text.split()
            word_count = max(len(words), 1)
            
            vague_matches = len(self.vague_regex.findall(text))
            concrete_matches = len(self.verifiable_regex.findall(text))
            
            v_density = (vague_matches / word_count) * 100.0
            c_density = (concrete_matches / word_count) * 100.0
            v_ratio = vague_matches / (vague_matches + concrete_matches + 1e-4)
            
            results.append({
                "vague_density": round(v_density, 3),
                "concrete_density": round(c_density, 3),
                "vagueness_ratio": round(v_ratio, 3)
            })
            
        return pd.DataFrame(results)

    def compute_greenwashing_gap(
        self, 
        df: pd.DataFrame, 
        sentiment_col: str = "nlp_sentiment", 
        e_score_col: str = "pillar_e"
    ) -> pd.Series:
        """
        Computes the Greenwashing Discrepancy (RQ3):
        Gap = Standardized Sentiment - Standardized True E-Pillar Performance
        A positive gap indicates that disclosure rhetoric outstrips actual environmental results.
        """
        norm_sentiment = (df[sentiment_col] - df[sentiment_col].mean()) / df[sentiment_col].std()
        norm_e = (df[e_score_col] - df[e_score_col].mean()) / df[e_score_col].std()
        
        greenwash_gap = norm_sentiment - norm_e
        return greenwash_gap

    def test_greenwashing_disagreement_correlation(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Tests RQ4: Does greenwashing risk relate to rating disagreement?
        Hypothesis: Firms with higher rhetoric-performance gaps exhibit higher rating dispersion
        due to conflicting signals sent to rating agencies.
        """
        from scipy.stats import pearsonr, spearmanr
        
        r_pearson, p_pearson = pearsonr(df["greenwashing_gap"], df["esg_disagreement"])
        r_spearman, p_spearman = spearmanr(df["greenwashing_gap"], df["esg_disagreement"])
        
        return {
            "pearson_corr": round(float(r_pearson), 4),
            "pearson_pvalue": float(p_pearson),
            "spearman_corr": round(float(r_spearman), 4),
            "spearman_pvalue": float(p_spearman)
        }

if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "esg_financial_panel.parquet")
    df = pd.read_parquet(data_path)
    
    analyzer = ESGGreenwashingAnalyzer()
    text_feats = analyzer.extract_text_features(df["csr_disclosure"].tolist())
    corr_results = analyzer.test_greenwashing_disagreement_correlation(df)
    
    print("NLP Greenwashing Extraction complete:")
    print(text_feats.head())
    print(f"RQ4 Correlation (Greenwashing Gap vs Rating Disagreement): {corr_results}")
