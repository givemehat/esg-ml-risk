"""
ESG & Financial Risk Machine Learning Research Pipeline
Data Loader and Multi-Agency Synthetic Engine
Based on Berg, Kölbel & Rigobon (2022), Avramov et al. (2022), and Raza et al. (2026).
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(name)s: %(message)s')

def generate_esg_financial_dataset(
    n_firms: int = 1500,
    years: list = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic multi-year panel of firm financials, multi-agency ESG ratings,
    and CSR textual disclosures with realistic Berg et al. (2022) divergence properties.
    """
    np.random.seed(seed)
    
    sectors = [
        "Energy", "Technology", "Healthcare", "Financials", 
        "Industrials", "Consumer Discretionary", "Materials", "Utilities"
    ]
    sector_weights = [0.10, 0.22, 0.15, 0.18, 0.15, 0.10, 0.05, 0.05]
    
    # Firm-level static traits
    firm_ids = [f"FIRM_{i:04d}" for i in range(n_firms)]
    firm_sectors = np.random.choice(sectors, size=n_firms, p=sector_weights)
    firm_sizes = np.random.lognormal(mean=8.5, sigma=1.2, size=n_firms) # Total Assets in $M
    firm_base_quality = np.random.normal(loc=0, scale=1.0, size=n_firms)
    
    records = []
    
    # CSR disclosure templates for greenwashing analysis
    csr_high_greenwash = [
        "We remain steadfastly committed to achieving net-zero emissions through world-leading eco-friendly paradigms and visionary climate stewardship.",
        "Our green innovation transition reflects an unmatched sustainability standard with transformative planetary impact and ethical harmony.",
        "We champion unparalleled circular economy breakthroughs, pioneering holistic environmental regeneration across our global supply ecosystem.",
        "Our comprehensive decarbonization vision leverages state-of-the-art restorative principles to safeguard environmental biodiversity."
    ]
    
    csr_low_greenwash = [
        "Scope 1 and 2 emissions fell 4.2% year-over-year to 1.12 MtCO2e, audited under ISO 14064-3 standards with verified carbon offsets.",
        "Independent third-party audits verified 78.4% hazardous waste diversion from landfills and reduced water withdrawal by 6.1 megaliters.",
        "Capital expenditure for renewable solar installations totaled $42.5 million, providing 34% of electricity consumption across manufacturing sites.",
        "Board oversight committee met quarterly to review safety incident frequency (0.42 per 200,000 hours worked) and employee turnover (8.5%)."
    ]
    
    csr_neutral = [
        "The company maintains established environmental and social guidelines compliant with regional statutory reporting directives.",
        "Governance policies continue to align with local market listings, executive compensation standards, and operational hazard reviews.",
        "Annual sustainability evaluations were conducted across operational units with regular progress tracking against corporate targets.",
        "Stakeholder engagement sessions were held across retail and corporate divisions regarding general workplace safety and compliance."
    ]

    for f_idx, f_id in enumerate(firm_ids):
        sec = firm_sectors[f_idx]
        base_size = firm_sizes[f_idx]
        base_q = firm_base_quality[f_idx]
        
        # Sector ESG baseline modifiers
        e_sector_mod = {"Energy": -12.0, "Utilities": -8.0, "Technology": +10.0, "Healthcare": +4.0}.get(sec, 0.0)
        distress_sector_mod = {"Energy": 0.03, "Financials": 0.02, "Technology": -0.02}.get(sec, 0.0)

        for year in years:
            time_trend = (year - 2018) * 0.8
            
            # 1. Financial Variables (Orbis / Compustat proxies)
            assets = base_size * np.exp(np.random.normal(0.04 * (year - 2018), 0.08))
            revenue = assets * np.random.uniform(0.6, 1.4)
            leverage = np.clip(np.random.beta(2, 5) + 0.15 - 0.05 * base_q + np.random.normal(0, 0.04), 0.05, 0.95)
            roa = np.clip(0.06 + 0.03 * base_q - 0.04 * leverage + np.random.normal(0, 0.03), -0.30, 0.35)
            current_ratio = np.clip(1.8 + 0.4 * base_q - 0.8 * leverage + np.random.normal(0, 0.25), 0.4, 5.0)
            operating_margin = np.clip(0.12 + 0.05 * base_q + np.random.normal(0, 0.04), -0.20, 0.45)
            debt_to_equity = leverage / max(1.0 - leverage, 0.05)
            tobins_q = np.clip(1.2 + 0.3 * roa + 0.2 * base_q + np.random.normal(0, 0.2), 0.4, 4.5)
            volatility = np.clip(0.28 - 0.08 * base_q + 0.15 * leverage + np.random.normal(0, 0.05), 0.10, 0.85)
            
            # Altman Z-Score proxy
            altman_z = (
                1.2 * (current_ratio - 1.0) / 2.0 +
                1.4 * roa +
                3.3 * operating_margin +
                0.6 * (1.0 / max(debt_to_equity, 0.1)) +
                1.0 * (revenue / max(assets, 1.0))
            )

            # 2. True Underlying ESG Construct (unobservable ground truth)
            true_e = np.clip(50.0 + 10.0 * base_q + e_sector_mod + time_trend + np.random.normal(0, 7.0), 5.0, 95.0)
            true_s = np.clip(52.0 + 8.0 * base_q + time_trend * 0.7 + np.random.normal(0, 6.0), 10.0, 95.0)
            true_g = np.clip(55.0 + 12.0 * base_q + np.random.normal(0, 5.0), 15.0, 95.0)
            true_esg = 0.35 * true_e + 0.35 * true_s + 0.30 * true_g

            # 3. Multi-Agency Divergence (Berg et al. 2022)
            # Measurement noise (56%), Scope difference (38%), Weight difference (6%)
            # Agency A (Proxy: Refinitiv / S&P: heavier E weight, operational metrics)
            agency_a = np.clip(
                0.45 * true_e + 0.30 * true_s + 0.25 * true_g +
                np.random.normal(0, 7.5), 0, 100
            )
            
            # Agency B (Proxy: MSCI: heavier risk exposure / governance)
            agency_b = np.clip(
                0.25 * true_e + 0.35 * true_s + 0.40 * true_g +
                np.random.normal(1.5, 8.2), 0, 100
            )
            
            # Agency C (Proxy: Sustainalytics: unmanaged risk focus)
            agency_c = np.clip(
                0.35 * true_e + 0.40 * true_s + 0.25 * true_g +
                np.random.normal(-1.0, 9.0), 0, 100
            )

            # Rating Disagreement (Standard deviation across the three agencies)
            agency_ratings = [agency_a, agency_b, agency_c]
            consensus_esg = float(np.mean(agency_ratings))
            esg_disagreement = float(np.std(agency_ratings))

            # 4. Text Disclosure & Greenwashing Index
            # Greenwashing = High positive corporate spin vs poor real environmental performance
            greenwashing_tendency = np.clip((70.0 - true_e) / 30.0 + np.random.normal(0, 0.4), 0.0, 2.5)
            
            if greenwashing_tendency > 1.2:
                text_snippet = np.random.choice(csr_high_greenwash)
                text_sentiment_score = np.random.uniform(0.75, 0.98) # Overly rosy
                text_vagueness_score = np.random.uniform(0.65, 0.95) # High vague claims
            elif greenwashing_tendency < 0.6:
                text_snippet = np.random.choice(csr_low_greenwash)
                text_sentiment_score = np.random.uniform(0.35, 0.65) # Measured, factual
                text_vagueness_score = np.random.uniform(0.10, 0.35) # High specific quantitative metrics
            else:
                text_snippet = np.random.choice(csr_neutral)
                text_sentiment_score = np.random.uniform(0.45, 0.70)
                text_vagueness_score = np.random.uniform(0.30, 0.60)

            # Greenwashing Gap = Sentiment vs Reality discrepancy
            normalized_real_e = true_e / 100.0
            greenwashing_gap = float(np.clip(text_sentiment_score - normalized_real_e, -1.0, 1.0))

            # 5. Financial Distress Probability in t+1 (Raza et al. 2026)
            # Logit model linking distress to financials + ESG levels + ESG disagreement + Greenwashing
            distress_logit = (
                -3.2
                - 1.1 * roa
                + 1.8 * leverage
                - 0.5 * np.log(assets)
                - 0.6 * (altman_z - 2.5)
                + 1.2 * volatility
                - 0.015 * consensus_esg     # ESG baseline effect
                + 0.045 * esg_disagreement  # Disagreement increases risk (Avramov et al. 2022)
                + 0.50 * max(0, greenwashing_gap) # Greenwashing penalty
                + distress_sector_mod
            )
            distress_prob = 1.0 / (1.0 + np.exp(-distress_logit))
            financial_distress_next_year = int(np.random.binomial(1, np.clip(distress_prob, 0.005, 0.85)))

            records.append({
                "firm_id": f_id,
                "year": year,
                "sector": sec,
                "assets": round(assets, 2),
                "log_assets": round(np.log(assets), 4),
                "revenue": round(revenue, 2),
                "leverage": round(leverage, 4),
                "roa": round(roa, 4),
                "current_ratio": round(current_ratio, 4),
                "operating_margin": round(operating_margin, 4),
                "debt_to_equity": round(debt_to_equity, 4),
                "tobins_q": round(tobins_q, 4),
                "volatility": round(volatility, 4),
                "altman_z": round(altman_z, 4),
                # ESG Pillar Scores
                "pillar_e": round(true_e, 2),
                "pillar_s": round(true_s, 2),
                "pillar_g": round(true_g, 2),
                "true_esg": round(true_esg, 2),
                # Multi-Agency Ratings
                "agency_refinitiv": round(agency_a, 2),
                "agency_msci": round(agency_b, 2),
                "agency_sustainalytics": round(agency_c, 2),
                "consensus_esg": round(consensus_esg, 2),
                "esg_disagreement": round(esg_disagreement, 2),
                # Text & Greenwashing
                "csr_disclosure": text_snippet,
                "nlp_sentiment": round(text_sentiment_score, 4),
                "nlp_vagueness": round(text_vagueness_score, 4),
                "greenwashing_gap": round(greenwashing_gap, 4),
                # Outcomes
                "distress_probability": round(distress_prob, 4),
                "financial_distress_t1": financial_distress_next_year
            })

    df = pd.DataFrame(records)
    return df

def get_temporal_splits(
    df: pd.DataFrame, 
    train_end_year: int = 2022
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict out-of-time train/test split.
    Train: years <= train_end_year (e.g. 2018-2022)
    Test: years > train_end_year (e.g. 2023-2025)
    Guarantees no future leakage as recommended by Raza et al. (2026).
    """
    train_df = df[df["year"] <= train_end_year].copy().reset_index(drop=True)
    test_df = df[df["year"] > train_end_year].copy().reset_index(drop=True)
    return train_df, test_df

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Generating comprehensive multi-agency ESG and financial distress panel...")
    df = generate_esg_financial_dataset(n_firms=1500, seed=42)
    csv_path = os.path.join(out_dir, "esg_financial_panel.parquet")
    df.to_parquet(csv_path, index=False)
    logger.info(f"Saved {len(df)} firm-year observations across {df['firm_id'].nunique()} firms to {csv_path}")
    train_df, test_df = get_temporal_splits(df)
    logger.info(f"Train set (2018-2022): {len(train_df)} rows")
    logger.info(f"Test set (2023-2025): {len(test_df)} rows")
