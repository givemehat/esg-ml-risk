"""
Dataset 2 Generator: UK Green Finance & TCFD Climate Disclosures Panel (2019-2025)
Based on:
- "Artificial Intelligence Models for Detecting Greenwashing in UK ESG Disclosures and Green Finance Projects" (2026)
- UK Streamlined Energy and Carbon Reporting (SECR) & Task Force on Climate-related Financial Disclosures (TCFD)
- London Stock Exchange (LSE) Green Bond & FTSE 350 proxies
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple

def generate_uk_green_finance_dataset(
    n_firms: int = 800,
    years: list = [2019, 2020, 2021, 2022, 2023, 2024, 2025],
    seed: int = 101
) -> pd.DataFrame:
    """
    Generates Dataset 2: Specialized UK Green Finance Panel.
    Tracks UK TCFD compliance, green bond issuances, borrowing spreads,
    audited SECR Scope 1-3 carbon emissions, and forward-looking commitment rhetoric.
    """
    np.random.seed(seed)
    
    uk_sectors = [
        "Financial Services & Banking", "Energy & Renewables", "Real Estate & Construction",
        "Manufacturing & Industrials", "Retail & Consumer", "Utilities & Water", "Technology & Telecom"
    ]
    sector_weights = [0.25, 0.12, 0.18, 0.15, 0.12, 0.08, 0.10]
    
    firm_ids = [f"UK_CORP_{i:04d}" for i in range(n_firms)]
    firm_sectors = np.random.choice(uk_sectors, size=n_firms, p=sector_weights)
    firm_assets_base = np.random.lognormal(mean=7.8, sigma=1.1, size=n_firms) # Assets in £M
    firm_green_integrity = np.random.normal(loc=0.0, scale=1.0, size=n_firms) # True integrity

    # Corporate TCFD disclosure statement templates
    tcfd_high_fls_vague = [
        "Our green finance framework aims to catalyze revolutionary, planet-positive ecological regeneration across British communities by 2050.",
        "We maintain visionary leadership in UK green bond markets, championing transformative circularity and aspirational climate harmony.",
        "Our sustainable finance roadmap aspires to pioneer holistic decarbonization through nature-aligned transition benchmarks.",
        "We are passionately committed to unlocking unprecedented environmental value through collaborative UK clean energy stewardship."
    ]
    
    tcfd_factual_audited = [
        "SECR compliant Scope 1 emissions totaled 412 ktCO2e (-8.4% YoY), independently assured by Deloitte under ISAE 3000 standards.",
        "Green bond proceeds of £250 million were 94.2% allocated to operational offshore wind farms delivering 1.2 TWh of clean power.",
        "Capital expenditure under the UK Green Taxonomy reached £84.5M, representing 42.1% of total corporate investment.",
        "Board climate committee tied 25% of executive Long-Term Incentive Plans (LTIP) to verified carbon reduction milestones."
    ]
    
    tcfd_neutral = [
        "The company submitted mandatory TCFD disclosures covering governance, strategy, risk management, and statutory emissions metrics.",
        "Climate risk assessments were conducted across operational commercial real estate assets following Bank of England stress scenarios.",
        "Corporate sustainability guidelines comply with UK Companies Act 2006 requirements and standard Financial Conduct Authority listing rules.",
        "Environmental compliance protocols were reviewed by the executive risk committee with annual carbon progress reports."
    ]

    records = []
    for f_idx, f_id in enumerate(firm_ids):
        sec = firm_sectors[f_idx]
        base_assets = firm_assets_base[f_idx]
        integrity = firm_green_integrity[f_idx]

        is_green_bond_issuer = int(np.random.binomial(1, p=np.clip(0.35 + 0.15 * integrity, 0.10, 0.80)))

        for year in years:
            time_trend = (year - 2019) * 1.1

            # Financial indicators (£M)
            assets = base_assets * np.exp(np.random.normal(0.03 * (year - 2019), 0.07))
            leverage = np.clip(0.42 - 0.08 * integrity + np.random.normal(0, 0.05), 0.08, 0.92)
            interest_coverage = np.clip(4.5 + 1.2 * integrity - 1.8 * leverage + np.random.normal(0, 0.4), 0.8, 15.0)
            ebitda_margin = np.clip(0.18 + 0.04 * integrity + np.random.normal(0, 0.03), -0.05, 0.45)
            
            # UK SECR Environmental Performance (0-100 scale)
            secr_carbon_performance = np.clip(54.0 + 12.0 * integrity + time_trend + np.random.normal(0, 6.0), 5.0, 98.0)
            tcfd_governance_score = np.clip(58.0 + 14.0 * integrity + np.random.normal(0, 5.0), 10.0, 98.0)
            green_capex_ratio = np.clip(0.20 + 0.15 * integrity + 0.02 * (year - 2019) + np.random.normal(0, 0.04), 0.01, 0.85)

            # Textual disclosure tendency
            greenwash_tendency = np.clip((65.0 - secr_carbon_performance) / 25.0 + 0.5 * leverage + np.random.normal(0, 0.3), 0.0, 3.0)
            
            if greenwash_tendency > 1.3:
                disclosure_text = np.random.choice(tcfd_high_fls_vague)
                sentiment_tone = np.random.uniform(0.78, 0.98)
                fls_vagueness = np.random.uniform(0.60, 0.95)
            elif greenwash_tendency < 0.7:
                disclosure_text = np.random.choice(tcfd_factual_audited)
                sentiment_tone = np.random.uniform(0.35, 0.60)
                fls_vagueness = np.random.uniform(0.10, 0.30)
            else:
                disclosure_text = np.random.choice(tcfd_neutral)
                sentiment_tone = np.random.uniform(0.45, 0.70)
                fls_vagueness = np.random.uniform(0.25, 0.55)

            # RDI Discrepancy
            rdi_gap = float(sentiment_tone - (secr_carbon_performance / 100.0))

            # UK Green Bond Spread / Credit Spread (in basis points, bps)
            # Higher greenwash gap and leverage penalize borrowing costs
            green_bond_spread_bps = float(np.clip(
                120.0 
                + 110.0 * leverage 
                - 15.0 * (interest_coverage - 4.0) 
                - 0.8 * secr_carbon_performance 
                + 45.0 * max(0, rdi_gap) 
                + np.random.normal(0, 12.0),
                45.0, 650.0
            ))

            # Financial impairment / Credit default in t+1
            distress_logit = (
                -3.8
                + 2.1 * leverage
                - 0.25 * interest_coverage
                - 0.40 * np.log(assets)
                + 0.005 * green_bond_spread_bps
                + 0.45 * max(0, rdi_gap)
            )
            prob_distress = 1.0 / (1.0 + np.exp(-distress_logit))
            distress_t1 = int(np.random.binomial(1, np.clip(prob_distress, 0.005, 0.85)))

            records.append({
                "firm_id": f_id,
                "year": year,
                "uk_sector": sec,
                "assets_gbp_m": round(assets, 2),
                "log_assets": round(np.log(assets), 4),
                "leverage": round(leverage, 4),
                "interest_coverage": round(interest_coverage, 4),
                "ebitda_margin": round(ebitda_margin, 4),
                "is_green_bond_issuer": is_green_bond_issuer,
                "green_capex_ratio": round(green_capex_ratio, 4),
                "secr_carbon_performance": round(secr_carbon_performance, 2),
                "tcfd_governance_score": round(tcfd_governance_score, 2),
                "tcfd_disclosure": disclosure_text,
                "nlp_sentiment": round(sentiment_tone, 4),
                "fls_vagueness": round(fls_vagueness, 4),
                "rdi_gap": round(rdi_gap, 4),
                "green_bond_spread_bps": round(green_bond_spread_bps, 2),
                "prob_distress": round(prob_distress, 4),
                "credit_distress_t1": distress_t1
            })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating Dataset 2: UK Green Finance & TCFD Climate Panel...")
    df_uk = generate_uk_green_finance_dataset(n_firms=800, seed=101)
    parquet_path = os.path.join(out_dir, "uk_tcfd_green_finance_panel.parquet")
    df_uk.to_parquet(parquet_path, index=False)
    print(f"Saved Dataset 2 with {len(df_uk)} firm-year observations across {df_uk['firm_id'].nunique()} UK firms to {parquet_path}")
    print(f"Years covered: {sorted(df_uk['year'].unique())}")
