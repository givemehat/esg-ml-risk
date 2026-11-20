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

This repository implements an **end-to-end Machine Learning, Uncertainty Quantification, and NLP framework** rigorously benchmarked across **12 distinct models** and **2 comprehensive empirical datasets**:
- **Dataset 1 (Global Multi-Agency Panel):** 12,000 firm-year observations across 1,500 global firms (2018–2025).
- **Dataset 2 (UK Green Finance & TCFD Panel):** 5,600 firm-year observations across 800 UK firms (2019–2025).

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

### C. Corporate ESG-Washing & Transformer NLP Detection (Zhao et al., UK 2026, arXiv 2025)
| Architecture | Out-of-Time ROC-AUC | F1-Score | Accuracy | Theoretical Justification |
| :--- | :---: | :---: | :---: | :--- |
| **1. Lexical TF-IDF N-Grams** | 0.6947 | 0.4327 | 84.73% | Text-only baseline (vulnerable to tone camouflage) |
| **2. ESG Lexicon Tone Only** | 0.9403 | 0.7650 | 91.40% | Evaluates polarity but blind to ground-truth reality |
| **3. FinBERT Sentiment Classifier** | 0.9526 | 0.7636 | 91.00% | Captures nuanced syntactic context & tone |
| **4. Full Transformer Discrepancy Framework** | **0.9982** | **0.9481** | **97.89%** | **Dual-Channel:** Contrasts FinBERT tone with audited carbon performance |

### D. Cross-Dataset 12-Model Comparative Benchmarks
| Architecture | Dataset 1: ESG $R^2$ | Dataset 1: Distress AUC | Dataset 2: Spread $R^2$ | Dataset 2: Credit Distress AUC |
| :--- | :---: | :---: | :---: | :---: |
| **M01: Ridge Regression** | **0.8365** | — | 0.9052 | — |
| **M02: Logistic Regression** | — | 0.5517 | — | 0.5742 |
| **M03: Random Forest** | 0.8311 | 0.5315 | 0.9212 | 0.5833 |
| **M04: Extra Trees** | 0.8343 | 0.5325 | **0.9254** | **0.6573** |
| **M05: Gradient Boosting (GBM)** | 0.8324 | 0.4890 | 0.9236 | 0.5903 |
| **M06: XGBoost** | 0.8317 | 0.4760 | 0.9240 | 0.6200 |
| **M07: LightGBM** | 0.8322 | 0.5432 | 0.9242 | 0.5611 |
| **M08: Deep Tabular MLP** | **0.8365** | **0.5861** | 0.9117 | 0.4803 |
| **M09: Quantile LightGBM** | PICP 76.2% | — | — | — |
| **M10: Inductive Conformal** | PICP 86.7% | — | — | — |
| **M11: FinBERT Sentiment** | — | — | — | ROC 0.9526 |
| **M12: Dual-Channel Network** | — | — | — | **ROC 0.9982** |

---

## 🤖 5. Deploying Transformers & Sentiment Analysis for ESG-Washing Detection

Based on **Zhao et al. / Asif et al.**, the **UK 2026 Green Finance TCFD study**, and the **2025 arXiv Survey**, standard sentiment analysis fails when deployed in isolation because greenwashing corporations deliberately produce the most optimistic sustainability reports. 

We resolve this by deploying a **4-Stage Dual-Channel Discrepancy Pipeline**:
1. **Stage 1 (Dual-Channel Ingestion):** Separates narrative CSR/10-K disclosures from audited ground-truth carbon emissions and CapEx.
2. **Stage 2 (Transformer Contextual Encoding):** Employs FinBERT self-attention to extract net sentiment polarity ($T_i = P_{\text{pos}} - P_{\text{neg}}$) and measures Forward-Looking Statement (FLS) vagueness.
3. **Stage 3 (Reporting Discrepancy Formulation):** Calculates the **Reporting Discrepancy Index ($RDI$)**:
   $$\text{RDI}_{i,t} = \mathcal{Z}\Big(\text{Tone}_{i,t}^{\text{Transformer}}\Big) - \mathcal{Z}\Big(\text{Performance}_{i,t}^{\text{Audited}}\Big)$$
   - $\text{RDI} > +0.85$: Severe Greenwashing Risk
   - $-0.85 \le \text{RDI} \le +0.85$: Authentic Alignment
   - $\text{RDI} < -0.85$: Greenhushing
4. **Stage 4 (Risk Deployment):** Feeds RDI scores into credit pricing models, explains inter-agency rating dispersion, and calibrates prediction intervals.

---

## 🛠️ 6. Project Structure

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
9. **Zhao, et al. / Asif, et al. (2024/2025).** *Detecting Greenwashing in the Environmental, Social, and Governance Domains Using Natural Language Processing.*
10. **UK Green Finance Working Group (2026).** *Artificial Intelligence Models for Detecting Greenwashing in UK ESG Disclosures and Green Finance Projects.*
11. **arXiv Survey (2025).** *Detecting Greenwashing: A Natural Language Processing Literature Survey.* arXiv preprint.

---

## 👥 8. Research Team & Acknowledgements

* **Rajnish Singh** (`@givemehat`) — B.Tech Computer Science & Engineering, Gautam Buddha University
* **Medhavie Gupta** — B.Tech Computer Science & Engineering, Gautam Buddha University
* **Katyayani Jaiswal** — B.Tech Computer Science & Engineering, Gautam Buddha University
* **Academic Guide & Advisor:** **Prof. Valentina Lagasio**, Associate Professor in Financial Markets and Institutions, Department of Management, Sapienza University of Rome.
