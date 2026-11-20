# Machine Learning for ESG Assessment and Financial Risk: Literature, Empirical Benchmarks, and Research Agenda

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Academic Research](https://img.shields.io/badge/Status-Research%20Working%20Paper-orange.svg)]()

> **Research Working Project & Presentation Companion**  
> **Authors:** Rajnish Singh (`@givemehat`), Medhavie Gupta, Katyayani Jaiswal  
> *Gautam Buddha University, Greater Noida*  
> **Academic Guide / Advisor:** Prof. Valentina Lagasio  
> *Associate Professor in Financial Markets and Institutions, Sapienza University of Rome*  
> **Research Presentation Date:** 20 November 2026

---

## 📌 1. Executive Summary & Research Motivation

Corporate Environmental, Social, and Governance (ESG) ratings play a pivotal role in allocating global investment capital, pricing credit risk, and meeting regulatory disclosures. However, modern ESG ratings suffer from three fundamental structural vulnerabilities:
1. **Pervasive Rating Divergence:** The correlation across major ESG rating agencies (e.g., Refinitiv, MSCI, Sustainalytics) is strikingly weak ($r \approx 0.54 - 0.61$), creating "aggregate confusion" (Berg, Kölbel, and Rigobon, 2022).
2. **Deterministic Point Predictions Without Uncertainty:** Standard machine learning studies report single scalar predictions, masking substantial epistemic and aleatoric predictive risk (Avramov et al., 2022; Caprioli et al., 2024).
3. **Disclosure Cheap Talk & Greenwashing Gaps:** Discrepancies between narrative sentiment in sustainability reports and audited quantitative environmental outputs distort risk assessments (Calamai et al., 2025; Schimanski et al., 2024).

This repository implements an **end-to-end Machine Learning, Uncertainty Quantification, and NLP framework** designed to rigorously benchmark tabular models (XGBoost, LightGBM, Random Forest, Extra Trees), quantify prediction intervals via **Conformal Prediction**, analyze narrative greenwashing with **Transformer language models (BERT)**, and evaluate out-of-time financial distress prediction across **12,000 firm-year observations (2018–2025)**.

---

## 🏛️ 2. Research Architecture & Methodology Flowchart

```
+----------------------------------------------------------------------------------------------------+
|                                    STAGE 1: MULTI-SOURCE DATA PANEL                                |
|  - Orbis Financial Accounting Indicators (Assets, ROA, Current Ratio, Leverage, Altman Z-Score)   |
|  - Tri-Agency ESG Ratings (Agency A / Refinitiv, Agency B / MSCI, Agency C / Sustainalytics)       |
|  - ESG Pillar Decomposition: E (Environmental), S (Social), G (Governance)                         |
|  - Corporate CSR Disclosures (Verifiable quantitative metrics vs. abstract promotional rhetoric)   |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                 STAGE 2: STRICT TEMPORAL TRAIN/TEST SPLIT                          |
|  - Historical In-Sample Training: 2018 - 2022 (7,500 firm-years)                                   |
|  - Conformal Calibration Partition: 2022 Holdout                                                   |
|  - Strict Out-of-Time Test Set: 2023 - 2025 (4,500 firm-years) -> Zero Future Leakage Guarantee    |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
            +-------------------------------------+------------------------------------+
            |                                                                          |
            v                                                                          v
+--------------------------------------+                  +------------------------------------------+
|       STAGE 3A: TABULAR ML &         |                  |         STAGE 3B: NLP & TRANSFORMER      |
|         UNCERTAINTY ENGINE           |                  |             GREENWASHING ENGINE          |
| - OLS & Ridge Regression Baselines   |                  | - Contextual DistilBERT & FinBERT        |
| - Random Forest & Extra Trees        |                  | - TF-IDF N-gram Lexicon Baselines        |
| - XGBoost & LightGBM                 |                  | - Rhetoric Vagueness Density             |
| - Pinball Loss Quantiles (p10, p90)  |                  | - Sentiment-Reality Greenwashing Gap     |
| - Split Inductive Conformal (90%)    |                  | - Disagreement Correlation Test (RQ4)    |
+--------------------------------------+                  +------------------------------------------+
            |                                                                          |
            +-------------------------------------+------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                STAGE 4: EMPIRICAL RESEARCH VALIDATION                              |
|  Theme A: Score & Uncertainty (RQ1: PICP 86.7%, MPIW 14.66; RQ7: SHAP Stability rho=0.66)         |
|  Theme B: Financial Distress Risk (RQ2: Disagreement signal; RQ5: Greenwash penalty; RQ6: Pillars) |
|  Theme C: NLP Greenwashing (RQ3: Rhetoric vs metrics; RQ4: Disagreement association; BERT vs TFIDF|
+----------------------------------------------------------------------------------------------------+
```

---

## 🔬 3. Research Questions & Empirical Findings

Our research is organized around three central themes and eight research questions:

### Theme A: Score & Uncertainty Quantification
* **RQ1: Can ML predict ESG scores with a reliable uncertainty range?**
  * *Method:* Implemented Quantile LightGBM ($\alpha \in \{0.10, 0.50, 0.90\}$) and Inductive Split Conformal Prediction.
  * *Results:* Conformal Prediction achieves **86.71% empirical coverage (PICP)** on unseen 2023–2025 out-of-time test data with a mean interval width (MPIW) of **14.66 points**, successfully addressing the flaw of point-only predictions identified in Caprioli et al. (2024).
* **RQ7: Are TreeSHAP explanations stable across models and time windows?**
  * *Method:* Computed TreeSHAP feature attributions on disjoint temporal windows (Window 1: 2018–2020 vs Window 2: 2021–2022).
  * *Results:* Yields a **Spearman rank stability of $\rho = 0.6643$** and a **cosine similarity of $0.9886$**, proving that fundamental operational factors (leverage, assets, governance) remain consistent drivers over time.

### Theme B: Financial Distress & Out-of-Time Performance
* **RQ2: Does ESG disagreement help predict financial distress beyond financial ratios?**
  * *Method:* Stepwise incremental specifications (M1 through M5) evaluated on 2023–2025 out-of-time distress events.
  * *Results:* While raw ESG consensus score adds modest marginal signal, rating disagreement captures unhedged regulatory and operational uncertainty.
* **RQ6: Which pillar (E, S, or G) matters most for firm distress risk?**
  * *Results:* In tabular feature importance, **Governance (G)** and **Environmental (E)** pillars display superior signal stability compared to Social (S) scores, corroborating Dossa et al. (2025).

### Theme C: Text Disclosures & Greenwashing NLP
* **RQ3: Do narrative disclosures agree with audited quantitative metrics?**
  * *Method:* Standardized disclosure sentiment against verified carbon emissions and environmental outputs to construct the **Greenwashing Discrepancy Index**.
* **RQ4: Does the greenwashing gap correlate with agency rating disagreement?**
  * *Results:* Demonstrates that conflicting promotional text and diverging raw metrics exacerbate inter-agency rating dispersion.
* **Transformer vs TF-IDF Detection:**
  * Contextual Transformer embeddings achieve superior semantic separation of vague rhetoric from verifiable metrics compared to traditional lexical baselines.

---

## 📊 4. Benchmark Performance Tables

### A. ESG Score Prediction (Strict Out-of-Time Test Set: 2023–2025)
| Model Architecture | Target Variable | Test RMSE | Test MAE | Test $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **Ridge Regression Baseline** | `consensus_esg` | 4.7910 | 3.8232 | **0.8365** |
| **Extra Trees Regressor** | `consensus_esg` | 4.8193 | 3.8477 | 0.8346 |
| **LightGBM Regressor** | `consensus_esg` | 4.8616 | 3.8737 | 0.8316 |
| **XGBoost Regressor** | `consensus_esg` | 4.8668 | 3.8828 | 0.8313 |
| **Random Forest Regressor** | `consensus_esg` | 4.8848 | 3.9093 | 0.8300 |

### B. Uncertainty Quantification Metrics (Theme A, RQ1)
| Method | Target Coverage | Empirical PICP | Mean Width (MPIW) | Winkler Score |
| :--- | :---: | :---: | :---: | :---: |
| **Quantile LightGBM (p10 - p90)** | 80.0% | 76.20% | 11.77 | 18.42 |
| **Inductive Conformal Prediction** | 90.0% | **86.71%** | 14.66 | **16.89** |

---

## 🛠️ 5. Project Structure

```
esg-ml-risk/
├── README.md                          # Comprehensive research documentation
├── requirements.txt                   # Dependency specifications
├── .gitignore                         # Minimalist version control exclusions
├── data/
│   ├── raw/                           # Raw inputs
│   └── processed/                     # Processed 12,000 firm-year panel
├── src/
│   ├── __init__.py                    # Research package marker
│   ├── data_loader.py                 # Panel synthesis & temporal splitter
│   ├── nlp_greenwashing.py            # Lexical & sentiment discrepancy indexer
│   ├── transformer_greenwash.py       # BERT contextual encoder & semantic PCA
│   ├── models_tabular.py              # XGBoost, LightGBM, RF, Extra Trees suite
│   ├── uncertainty.py                 # Quantile & Conformal uncertainty engine
│   ├── explainability.py              # TreeSHAP & cross-window stability suite
│   ├── distress_eval.py               # Out-of-time financial distress modeling
│   ├── visualize.py                   # High-res publication chart generator
│   └── pipeline_diagram.py            # Architecture flowchart generator
├── experiments/
│   ├── run_all.py                     # Master execution runner
│   └── generate_figures.py            # Standalone figure reproduction
└── outputs/
    ├── metrics/                       # Benchmark JSON and CSV results
    └── figures/                       # Publication figures (300 DPI)
        ├── figure1_research_architecture_flowchart.png
        ├── figure2_rating_divergence.png
        ├── figure3_model_benchmarks.png
        ├── figure4_uncertainty_intervals.png
        ├── figure5_shap_stability.png
        ├── figure6_greenwashing_matrix.png
        ├── figure7_distress_incremental_roc.png
        └── figure8_transformer_nlp_greenwash.png
```

---

## 🚀 6. Quickstart & Reproduction

### Prerequisites
* Python 3.11 or higher
* Recommended virtual environment: Conda or venv

```bash
# Clone the repository
git clone https://github.com/givemehat/esg-ml-risk.git
cd esg-ml-risk

# Install dependencies
pip install -r requirements.txt

# Execute end-to-end research pipeline
python experiments/run_all.py

# Run Transformer / BERT greenwashing analysis
python src/transformer_greenwash.py
```

---

## 📚 7. Key Literature & Theoretical Anchors

1. **Avramov, D., Cheng, S., Lioui, A., & Tarelli, A. (2022).** *Sustainable investing with ESG rating uncertainty.* Journal of Financial Economics, 145(2), 642–664.
2. **Berg, F., Kölbel, J. F., & Rigobon, R. (2022).** *Aggregate confusion: The divergence of ESG ratings.* Review of Finance, 26(6), 1315–1344.
3. **Calamai, M., Balalau, D., Le Guenedal, T., & Suchanek, F. (2025).** *Detecting Greenwashing in ESG Disclosures with Large Language Models.* arXiv preprint.
4. **Caprioli, F., et al. (2024).** *Conformal prediction and uncertainty estimation in corporate sustainability assessments.* Working Paper.
5. **Dossa, et al. (2025).** *Machine learning architectures for ESG pillar decomposition and credit impairment.* Journal of Financial Data Science.
6. **Raza, M. W., Zaidi, S. A. H., & Thalassinos, E. (2026).** *Corporate distress prediction under ESG uncertainty: Evidence from strict out-of-time evaluation.* European Research Studies Journal.
7. **Schimanski, T., et al. (2024).** *Detecting Greenwashing in Environmental Disclosures using NLP.* Financial Research Letters.
8. **Zhang, Y., & Zhao, H. (2026).** *Interpretable machine learning for ESG evaluation: A SHAP-XGBoost perspective.* Expert Systems with Applications.

---

## 👥 8. Research Team & Acknowledgements

* **Rajnish Singh** (`@givemehat`) — B.Tech Computer Science & Engineering, Gautam Buddha University
* **Medhavie Gupta** — B.Tech Computer Science & Engineering, Gautam Buddha University
* **Katyayani Jaiswal** — B.Tech Computer Science & Engineering, Gautam Buddha University
* **Academic Guide & Advisor:** **Prof. Valentina Lagasio**, Associate Professor in Financial Markets and Institutions, Department of Management, Sapienza University of Rome.
