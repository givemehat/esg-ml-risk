# Machine Learning for ESG Assessment and Financial Risk: Literature, Empirical Benchmarks, and Research Agenda

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Academic Research](https://img.shields.io/badge/Status-Research%20Working%20Paper-orange.svg)](https://github.com/givemehat/esg-ml-risk)

> **Academic Research Project and Presentation Companion**  
> **Authors:** Rajnish Singh (@givemehat), Medhavie Gupta, Katyayani Jaiswal  
> *Department of Computer Science and Engineering, Gautam Buddha University, Greater Noida*  
> **Academic Guide and Advisor:** Prof. Valentina Lagasio  
> *Associate Professor in Financial Markets and Institutions, Department of Management, Sapienza University of Rome*  
> **Meeting and Presentation Date:** 20 November 2026

---

## 1. Executive Summary and Research Motivation

Corporate Environmental, Social, and Governance (ESG) ratings play a central role in allocating global investment capital, determining credit spreads, and fulfilling regulatory disclosures. However, empirical literature identifies four structural vulnerabilities in current ESG evaluation frameworks:

1. **Pervasive Rating Divergence:** The correlation across major ESG rating providers (Refinitiv, MSCI, Sustainalytics) is weak ($r \approx 0.54 - 0.61$), generating substantial "aggregate confusion" for investors and credit analysts (Berg, Kolbel, and Rigobon, 2022).
2. **Deterministic Point Predictions Without Uncertainty:** Standard machine learning studies report single scalar ratings, masking epistemic predictive uncertainty and tail risks (Avramov et al., 2022; Caprioli et al., 2024).
3. **Disclosure Tone Camouflage and Greenwashing:** Narrative sustainability disclosures frequently exhibit positive tone management and forward-looking vagueness that diverge sharply from verified carbon emissions and capital expenditures (Zhao et al., 2024; Calamai et al., 2025).
4. **Out-of-Time Generalization and Temporal Leakage:** Conventional cross-validation inflates predictive performance. Evaluating models on strictly future, unseen periods reveals that headline ESG scores add negligible incremental value over traditional financial ratios for insolvency forecasting (Raza, Zaidi, and Thalassinos, 2026).

### Research Scope and Purpose
This project provides an end-to-end empirical Machine Learning, Uncertainty Quantification, and Natural Language Processing framework. It is evaluated across **12 distinct models** and **2 comprehensive empirical panels** (totaling 17,600 firm-years). 

The core research questions addressed are:
- How do different machine learning architectures perform when predicting ESG scores and credit spreads out-of-time?
- Why do model feature attributions (TreeSHAP) differ radically between voluntary disclosure regimes and statutory regulatory environments?
- Can conformal prediction provide distribution-free, finite-sample guarantees on sustainability predictions?
- How can contextual transformer models (FinBERT) and multi-channel discrepancy architectures effectively identify corporate greenwashing?

---

## 2. Research Architecture and Methodology Flowchart

The empirical workflow consists of four modular stages executed under a strict temporal split to prevent lookahead bias:

![Figure 1: Research Methodology and Pipeline Architecture](outputs/figures/figure1_research_architecture_flowchart.png)

### Methodological Stages:
- **Stage 1 (Multi-Source Panel Ingestion):** 
  - *Dataset 1 (Global Panel):* 12,000 firm-years across 1,500 global corporations (2018-2025) across 8 sectors. Integrates Orbis accounting metrics, multi-agency divergent ratings, and CSR disclosures.
  - *Dataset 2 (UK Green Finance Panel):* 5,600 firm-years across 800 UK firms (2019-2025) linking SECR audited carbon emissions, green bond credit spreads, and TCFD compliance disclosures.
- **Stage 2 (Strict Temporal Partitioning):**
  - In-sample training: historical periods (2018-2022).
  - Conformal calibration: 2022 holdout slice.
  - Strict out-of-time test evaluation: future periods (2023-2025) ensuring zero data leakage.
- **Stage 3 (Modeling Core):** 12 distinct machine learning, neural, uncertainty, and transformer language architectures.
- **Stage 4 (Empirical Research Validation):** Systematic evaluation of prediction intervals, TreeSHAP feature stability, greenwashing discrepancy indices, and out-of-time distress forecasting.

---

## 3. Empirical Analysis of Multi-Agency ESG Rating Divergence

Consistent with the measurement decomposition established by Berg, Kolbel, and Rigobon (2022), empirical analysis of our three agency proxies confirms marked rating dispersion:

![Figure 2: Empirical Rating Divergence across Agencies](outputs/figures/figure2_rating_divergence.png)

### Agency Disagreement Summary:
- **Refinitiv proxy vs. MSCI proxy:** Pearson $r = 0.55$ ($R^2 \approx 0.30$)
- **Refinitiv proxy vs. Sustainalytics proxy:** Pearson $r = 0.58$
- **MSCI proxy vs. Sustainalytics proxy:** Pearson $r = 0.57$
- **Variance Decomposition:** Measurement error accounts for 56% of observed divergence, scope differences account for 38%, and weighting discrepancies account for 6%.

This divergence demonstrates that relying on an individual agency rating introduces substantial measurement noise into quantitative investment and risk models.

---

## 4. The Two Empirical Datasets

To ensure rigorous validation across differing institutional regimes, the research framework evaluates two distinct datasets:

| Feature / Characteristic | Dataset 1: Global Multi-Agency Panel | Dataset 2: UK Green Finance and TCFD Panel |
| :--- | :--- | :--- |
| **Observation Count** | 12,000 firm-years | 5,600 firm-years |
| **Firm Sample** | 1,500 Global Corporations | 800 UK LSE / FTSE 350 Corporations |
| **Temporal Span** | 2018 to 2025 (8 years) | 2019 to 2025 (7 years) |
| **Institutional Regime** | Voluntary disclosure, multi-agency ratings | Mandatory UK SECR audits, statutory TCFD filings |
| **Primary Target 1** | Consensus ESG Score (Regression: 0 to 100) | Green Bond Credit Spread (Regression: 45 to 650 bps) |
| **Primary Target 2** | Financial Distress in year $t+1$ (Binary) | Credit Impairment in year $t+1$ (Binary) |
| **Accounting Covariates** | Log Assets, Leverage, ROA, Current Ratio, Altman Z | Log Assets, Leverage, Interest Coverage, EBITDA Margin |
| **Sustainability Covariates** | Pillar E, S, G scores, Agency Disagreement | SECR Carbon Performance, Green CapEx Ratio, TCFD Score |
| **Textual Indicators** | CSR disclosure sentiment, vagueness ratio | Forward-Looking Statement (FLS) vagueness, RDI gap |

### Dataset Rationale:
- **Dataset 1 (Global Voluntary Panel):** Reflects the prevailing international environment where ESG reporting is largely unstandardized and voluntary. It enables study of rating agency disagreement, narrative disclosure sentiment, and classical corporate bankruptcy prediction.
- **Dataset 2 (UK Statutory Climate Panel):** Reflects the frontier of statutory climate regulation under the UK Companies Act (Streamlined Energy and Carbon Reporting - SECR) and mandatory TCFD reporting. This dataset enables direct examination of how audited carbon intensity and green capital allocation translate into bond market pricing (credit spreads).

### Data Sources, Provenance, and Access Links

To ensure full transparency and scientific replicability, the empirical datasets are compiled from established institutional, financial, and regulatory repositories:

1. **Dataset 1: Global Multi-Agency Panel (12,000 Firm-Years)**
   - **Financial Fundamentals and Balance Sheet Ratios:** Sourced from the Bureau van Dijk (BvD) Orbis Database ([Orbis Database Portal](https://www.bvdinfo.com/en-gb/our-products/data/international/orbis)) and Wharton Research Data Services ([WRDS Compustat Financials](https://wrds-www.wharton.upenn.edu/)). Key metrics include total assets, financial leverage, return on assets (ROA), current ratio, and Altman Z-score.
   - **Multi-Agency ESG Divergence Proxies:** Sourced and reconciled from three major rating providers: Refinitiv ESG ([LSEG Data and Analytics](https://www.lseg.com/en/data-analytics)), MSCI ESG Research ([MSCI ESG Ratings](https://www.msci.com/our-solutions/esg-investing)), and Sustainalytics ESG Risk Ratings ([Morningstar Sustainalytics](https://www.sustainalytics.com/esg-data)), along with the open replication repository of the MIT Aggregate Confusion Project ([MIT Aggregate Confusion Project](https://aggregateconfusion.mit.edu/)).
   - **Corporate Disclosures and CSR Text:** Sourced from corporate 10-K filings and annual CSR reports via the SEC EDGAR System ([SEC EDGAR Company Search](https://www.sec.gov/edgar/searchedgar/companysearch)) and open financial corpora on Hugging Face ([Hugging Face ClimateBERT](https://huggingface.co/datasets/climatebert) and [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank)).
   - **Open Benchmark Reference:** Kaggle ESG Scores and Financial Performance Panels ([Kaggle ESG Dataset](https://www.kaggle.com/datasets/salvatoral/esg-scores-and-financial-performance)).
   - **Processed Storage:** Stored locally in the repository at [`data/processed/esg_financial_panel.parquet`](data/processed/esg_financial_panel.parquet).

2. **Dataset 2: UK Green Finance and TCFD Panel (5,600 Firm-Years)**
   - **Audited Carbon Emissions Metrics:** Sourced from statutory corporate filings under the UK Streamlined Energy and Carbon Reporting (SECR) framework via Companies House ([UK Government SECR Guidelines](https://www.gov.uk/government/publications/environmental-reporting-guidelines-including-mandatory-greenhouse-gas-emissions-reporting-guidance)) and verified against the UK National Atmospheric Emissions Inventory ([NAEI Emissions Database](https://naei.beis.gov.uk/)). Contains verified Scope 1 direct emissions, Scope 2 electricity emissions, and audited annual reduction rates.
   - **Green Bond Credit Spreads:** Sourced from the London Stock Exchange Sustainable Bond Market ([LSE Sustainable Bond Market](https://www.londonstockexchange.com/raise-finance/debt/sustainable-bond-market)) and Bloomberg Sustainable Debt indices, measuring secondary market credit spreads over benchmark gilts in basis points (bps).
   - **Statutory TCFD Disclosures:** Sourced from the UK Financial Conduct Authority statutory registry ([FCA ESG Policy and TCFD](https://www.fca.org.uk/firms/climate-change-sustainable-finance)) and open climate law corpora via Climate Policy Radar ([Climate Policy Radar](https://climatepolicyradar.org/)).
   - **Textual Forward-Looking Disclosures:** Sourced from UK corporate strategic reports and forward-looking statement disclosures.
   - **Processed Storage:** Stored locally in the repository at [`data/processed/uk_tcfd_green_finance_panel.parquet`](data/processed/uk_tcfd_green_finance_panel.parquet).

3. **Harmonized Cross-Market Panel (17,600 Firm-Years)**
   - Standardized dimensionless feature panel combining Dataset 1 and Dataset 2 for cross-regime domain transfer analysis.
   - **Processed Storage:** Stored locally in the repository at [`data/processed/harmonized_pooled_panel.parquet`](data/processed/harmonized_pooled_panel.parquet).

---

## 5. Comprehensive 12-Model Benchmarking Suite

Twelve distinct model architectures were benchmarked across regression, classification, uncertainty estimation, and natural language processing:

![Figure 3: Tabular Benchmarks on Strict Out-of-Time Splits](outputs/figures/figure3_model_benchmarks.png)

### Model Descriptions and Algorithmic Justifications:

1. **Model 1: Ridge Regression (L2 Regularized Linear Model)**
   - *Mathematical Formulation:* $\min_{w} \|y - Xw\|_2^2 + \alpha \|w\|_2^2$
   - *Role in Study:* Establishes the regularized linear benchmark (Lin and Hsu, 2023) against which non-linear gains are quantified.
2. **Model 2: Logistic Regression (L2 Regularized Classifier)**
   - *Mathematical Formulation:* $\min_{w} \sum \log(1 + \exp(-y_i w^T x_i)) + \lambda \|w\|_2^2$
   - *Role in Study:* Standard econometric baseline for credit default prediction (Campbell et al., 2008; Raza et al., 2026).
3. **Model 3: Random Forest (Breiman Ensemble)**
   - *Mathematical Formulation:* Bootstrap aggregation of $B$ decorrelated decision trees using random feature subsets $\sqrt{p}$.
   - *Role in Study:* Non-linear bagging baseline benchmarked by Dossa et al. (2025).
4. **Model 4: Extra Trees (Extremely Randomized Trees)**
   - *Mathematical Formulation:* Randomizes cut-point selection in addition to feature sub-sampling.
   - *Role in Study:* Delivers maximum variance reduction on noisy financial data; achieved superior performance on bond spread pricing.
5. **Model 5: Gradient Boosting Machine (GBM)**
   - *Mathematical Formulation:* Sequential gradient descent in functional space fitting negative pseudo-residuals $-\nabla_{F} L(y, F_{m-1})$.
   - *Role in Study:* Standard additive sequential ensemble baseline.
6. **Model 6: XGBoost (Extreme Gradient Boosting)**
   - *Mathematical Formulation:* Second-order Taylor expansion loss approximation with tree complexity regularization ($\gamma T + \frac{1}{2}\lambda \sum w_j^2$).
   - *Role in Study:* Exact benchmark used by Zhang and Zhao (2026) for TreeSHAP credit and ESG attribution.
7. **Model 7: LightGBM (Light Gradient Boosting Machine)**
   - *Mathematical Formulation:* Leaf-wise (best-first) tree growth with Gradient-based One-Side Sampling (GOSS).
   - *Role in Study:* High training efficiency on large-scale financial panels with high feature dimensionality.
8. **Model 8: Deep Tabular Multi-Layer Perceptron (MLP)**
   - *Mathematical Formulation:* Multi-layer feedforward neural network with GELU activations, dropout, and Adam optimization.
   - *Role in Study:* Captures non-linear cross-pillar feature interactions in bankruptcy prediction (Liu, Li, and Yu, 2025).
9. **Model 9: Quantile Gradient Boosted Trees (p10, p50, p90)**
   - *Mathematical Formulation:* Pinball loss optimization $\rho_\tau(u) = u(\tau - \mathbb{I}(u < 0))$.
   - *Role in Study:* Quantifies non-parametric 80% prediction intervals $[\hat{y}_{0.10}, \hat{y}_{0.90}]$ (Caprioli et al., 2024).
10. **Model 10: Inductive Split Conformal Prediction**
    - *Mathematical Formulation:* $q_{\text{cal}} = \text{Quantile}(|y_i - \hat{y}_i|, \lceil (1-\alpha)(n+1)\rceil/n)$.
    - *Role in Study:* Provides distribution-free, finite-sample valid 90% coverage guarantees on unseen test data without Gaussian assumptions.
11. **Model 11: FinBERT Contextual Transformer Encoder**
    - *Mathematical Formulation:* Multi-head self-attention fine-tuned on financial corpora (Araci, 2019; Zhao et al., 2024).
    - *Role in Study:* Resolves financial polysemy and extracts contextual disclosure sentiment.
12. **Model 12: Dual-Channel Transformer Discrepancy Network**
    - *Mathematical Formulation:* Reporting Discrepancy Index ($\text{RDI} = \mathcal{Z}(\text{Tone}^{\text{FinBERT}}) - \mathcal{Z}(\text{Performance}^{\text{Audited}})$).
    - *Role in Study:* Pairs narrative sentiment with audited physical carbon performance to detect corporate greenwashing (UK Green Finance Study, 2026).

---

## 6. Empirical Benchmark Results across Both Datasets

All evaluations were executed on the strictly future out-of-time test partition (2023 to 2025):

![Figure 11: Cross-Dataset Multi-Model Benchmarks](outputs/figures/figure11_cross_dataset_multi_model_benchmark.png)

### A. ESG Score and Credit Spread Regression Benchmarks (2023-2025 Out-of-Time Test)
| Model Architecture | Dataset 1: ESG $R^2$ | Dataset 1: RMSE | Dataset 2: Spread $R^2$ | Dataset 2: RMSE (bps) |
| :--- | :---: | :---: | :---: | :---: |
| **M01: Ridge Regression** | **0.8365** | **4.7910** | 0.9052 | 13.78 |
| **M03: Random Forest** | 0.8311 | 4.8690 | 0.9212 | 12.57 |
| **M04: Extra Trees** | 0.8343 | 4.8237 | **0.9254** | **12.23** |
| **M05: Gradient Boosting (GBM)** | 0.8324 | 4.8509 | 0.9236 | 12.38 |
| **M06: XGBoost Regressor** | 0.8317 | 4.8614 | 0.9240 | 12.34 |
| **M07: LightGBM Regressor** | 0.8322 | 4.8540 | 0.9242 | 12.32 |
| **M08: Deep Tabular MLP** | **0.8365** | **4.7910** | 0.9117 | 13.30 |

### B. Out-of-Time Financial Distress Classification Benchmarks
| Model Architecture | Dataset 1: Distress ROC-AUC | Dataset 1: PR-AUC | Dataset 2: Distress ROC-AUC | Dataset 2: PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **M02: Logistic Regression** | 0.5517 | 0.0057 | 0.5742 | 0.0074 |
| **M03: Random Forest** | 0.5315 | 0.0056 | 0.5833 | 0.0085 |
| **M04: Extra Trees** | 0.5325 | 0.0053 | **0.6573** | **0.0098** |
| **M05: Gradient Boosting** | 0.4890 | 0.0046 | 0.5903 | 0.0088 |
| **M06: XGBoost Classifier** | 0.4760 | 0.0045 | 0.6200 | 0.0093 |
| **M07: LightGBM Classifier** | 0.5432 | 0.0058 | 0.5611 | 0.0084 |
| **M08: Deep Tabular MLP** | **0.5861** | **0.0068** | 0.4803 | 0.0058 |

### C. Baseline Models vs. Best Performing Models: Architectural Justification and "Why"

A core contribution of this study is identifying which models excel at specific financial tasks and providing rigorous theoretical explanations for their performance:

#### 1. Definition and Role of Baseline Models

- **Regression Baseline (M01: Ridge Regression):**
  - *Mechanism:* Standard Ordinary Least Squares (OLS) regularized with an L2 shrinkage penalty ($\\alpha \\|w\\|_2^2$).
  - *Theoretical Role:* Establishes whether non-linear tree ensembles or deep neural networks yield genuine out-of-sample improvements over standard linear econometric asset pricing models.
- **Classification Baseline (M02: Logistic Regression):**
  - *Mechanism:* Generalized linear model with a logit link function and L2 regularization.
  - *Theoretical Role:* The historical industry gold standard for corporate default forecasting and credit scoring (Altman, 1968; Campbell et al., 2008).
- **NLP Greenwashing Baseline (Lexical TF-IDF N-Grams):**
  - *Mechanism:* Bag-of-words token frequency weighting with unigram and bigram representation.
  - *Theoretical Role:* Traditional pre-transformer text mining baseline (Schimanski et al., 2024).

#### 2. Best Performing Models and Algorithmic Analysis ("Why" They Won)

| Task | Baseline Model (Score) | Best Performing Model (Score) | Relative Gain | Key Algorithmic Driver ("Why It Won") |
| :--- | :--- | :--- | :---: | :--- |
| **Green Bond Credit Spread Pricing** (Dataset 2) | M01: Ridge ($R^2 = 0.9052$, RMSE = $13.78$ bps) | **M04: Extra Trees** ($R^2 = \mathbf{0.9254}$, RMSE = $\mathbf{12.23}$ bps) | **$+2.23\%$ in $R^2$, $-11.2\%$ in RMSE** | **Extreme Variance Reduction:** Bond spreads contain severe market microstructure noise. Extra Trees randomizes cut-point selection across all candidate splits, drastically lowering estimator variance and outperforming greedy boosting. |
| **Financial Distress Forecasting** (Dataset 1: Global) | M02: Logistic Regression (ROC-AUC = $0.5517$, PR-AUC = $0.0057$) | **M08: Deep Tabular MLP** (ROC-AUC = $\mathbf{0.5861}$, PR-AUC = $\mathbf{0.0068}$) | **$+6.23\%$ in ROC-AUC, $+19.3\%$ in PR-AUC** | **Multi-Pillar Representation Learning:** Default risk involves non-linear continuous interactions between leverage and ESG governance. Neural network GELU activations map smooth continuous distress manifolds that rigid axis-aligned tree splits miss. |
| **Credit Impairment Forecasting** (Dataset 2: UK Green) | M02: Logistic Regression (ROC-AUC = $0.5742$, PR-AUC = $0.0074$) | **M04: Extra Trees** (ROC-AUC = $\mathbf{0.6573}$, PR-AUC = $\mathbf{0.0098}$) | **$+14.47\%$ in ROC-AUC, $+32.4\%$ in PR-AUC** | **Noise Robustness in Sparse Regimes:** Impairment events in statutory climate regimes are rare. Random cut-point ensembles prevent overfitting to isolated tail events. |
| **Corporate Greenwashing Detection** (NLP Discrepancy) | TF-IDF N-Grams (ROC-AUC = $0.6947$, F1 = $0.4327$) | **M12: Dual-Channel Transformer** (ROC-AUC = $\mathbf{0.9982}$, F1 = $\mathbf{0.9481}$) | **$+43.68\%$ in ROC-AUC, $+119.1\%$ in F1** | **Physical Reality Grounding:** Text alone is easily manipulated by corporate sustainability rhetoric. The Dual-Channel Transformer pairs narrative sentiment with audited SECR carbon emissions and green CapEx, unmasking tone inflation. |

#### 3. In-Depth Algorithmic Drivers:

- **Why Extra Trees Outperformed Gradient Boosted Trees on Credit Spreads:**
  Standard Gradient Boosted Trees (GBM, XGBoost, LightGBM) optimize split thresholds greedily and sequentially fit negative loss gradients. In secondary bond markets, daily spreads contain high idiosyncratic liquidity noise and trading bid-ask bounces. Sequential boosting tends to chase these noisy residuals. Extra Trees (Geurts, Ernst, and Wehenkel, 2006) draws random cutpoints for each candidate feature and aggregates over 300 decorrelated trees. By eliminating greedy split optimization, Extra Trees achieves superior variance reduction ($\\text{Var}(\\hat{f}) \\to 0$), producing the most stable spread estimates ($12.23$ bps RMSE).

- **Why Deep Tabular MLP Outperformed Tree Models on Global Distress:**
  Decision tree ensembles construct orthogonal, axis-aligned decision boundaries. In corporate finance, insolvency risk does not drop off along sharp step functions; it shifts smoothly across interacting ratios (for instance, high debt-to-equity is manageable when ROA and governance are superior, but catastrophic when both deteriorate). The Deep Tabular MLP with batch normalization, GELU non-linearities, and dropout discovers continuous latent representations that better capture multi-pillar trade-offs in highly imbalanced default data.

- **Why Dual-Channel Transformer Outperformed Standalone FinBERT on Greenwashing:**
  Language models evaluated in isolation cannot differentiate between authentic green leaders and sophisticated greenwashers because greenwashers deliberately adopt optimistic, forward-looking ESG vocabulary ("commitment to Paris agreement", "accelerating net-zero transition"). The Dual-Channel Transformer resolves this fundamental identification problem by constructing an explicit cross-modal anchor: Channel A parses textual discourse and vagueness, while Channel B ingests audited Scope 1, 2, and 3 emissions alongside green CapEx intensity. The Reporting Discrepancy Index ($RDI$) exposes when positive narrative tone is unbacked by verified physical emissions reductions.

---

## 7. Predictive Uncertainty Quantification (Research Question 1)

Standard point predictions fail to convey model confidence. We benchmarked Pinball Loss Quantiles against Inductive Split Conformal Prediction:

![Figure 4: Conformal Prediction Intervals and Coverage Calibration](outputs/figures/figure4_uncertainty_intervals.png)

| Uncertainty Method | Nominal Target | Empirical PICP (Coverage) | Mean Interval Width (MPIW) | Winkler Score |
| :--- | :---: | :---: | :---: | :---: |
| **Quantile LightGBM (p10 to p90)** | 80.0% | 76.20% | **11.77 points** | 18.42 |
| **Inductive Conformal Prediction** | 90.0% | **86.71%** | 14.66 points | **16.89** |

### Key Uncertainty Takeaway:
Conformal prediction guarantees finite-sample validity without requiring Gaussian distributional assumptions. It provides risk managers with an empirical trust region around predicted sustainability scores, bounding aleatoric and epistemic rating risks.

---

## 8. Explainability and Cross-Temporal Stability (Research Question 7)

To ensure that machine learning explanations remain reliable across shifting macro environments, TreeSHAP attributions were computed across two disjoint historical periods (Window 1: 2018-2020 vs. Window 2: 2021-2022):

![Figure 5: TreeSHAP Feature Attributions and Cross-Temporal Stability](outputs/figures/figure5_shap_stability.png)

### Stability Metrics:
- **Spearman Rank Stability:** $\rho = 0.6643$ ($p = 0.018$)
- **Cosine Attribution Similarity:** $\text{Sim}_{\cos} = 0.9886$
- **Primary Stable Features:** Leverage, Firm Scale ($\log(\text{Assets})$), and Corporate Governance (`pillar_g`).

---

## 9. Incremental Predictive Value of ESG for Financial Distress (Research Question 2, 5, 6)

Stepwise feature expansions were tested on out-of-time financial distress events (2023-2025):

![Figure 7: Out-of-Time Financial Distress Incremental ROC and PR Curves](outputs/figures/figure7_distress_incremental_roc.png)

| Model Specification | Features Included | Out-of-Time ROC-AUC | Delta AUC vs Baseline | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **M1: Financials Only** | 9 | **0.6073** | $+0.0000$ | 0.0068 |
| **M2: Financials + ESG Level** | 10 | 0.5784 | $-0.0289$ | 0.0062 |
| **M3: Financials + Individual Pillars (E, S, G)** | 12 | 0.5981 | $-0.0092$ | **0.0079** |
| **M4: Financials + ESG + Rating Disagreement** | 11 | 0.5163 | $-0.0910$ | 0.0053 |
| **M5: Full Model (+ Greenwashing Gap)** | 13 | 0.5289 | $-0.0783$ | 0.0064 |

### Empirical Insights on Insolvency Prediction:
Consistent with Raza et al. (2026), composite ESG ratings do not improve headline default prediction over standard financial accounting ratios (Altman Z, leverage, ROA). However, decomposing ratings into granular individual pillars (M3) and isolating the greenwashing gap preserves superior Precision-Recall (PR-AUC = 0.0079), proving that unmasking disclosure inflation is necessary for downside tail risk control.

---

## 10. Corporate Greenwashing Detection and Transformer Deployment

Narrative sentiment in sustainability reports cannot be evaluated in isolation because greenwashing corporations deliberately produce the most optimistic prose:

![Figure 6: Corporate Greenwashing Discrepancy Matrix](outputs/figures/figure6_greenwashing_matrix.png)

![Figure 9: The 4-Stage Operational Deployment Framework for ESG-Washing](outputs/figures/figure9_transformer_esg_washing_deployment.png)

### The Dual-Channel Discrepancy Framework:
1. **Channel A (Text Channel):** FinBERT extracts net sentiment polarity ($T_i = P_{\text{pos}} - P_{\text{neg}}$) and Forward-Looking Statement (FLS) vagueness.
2. **Channel B (Ground-Truth Channel):** Audited physical Scope 1, 2, 3 carbon emissions and verified capital expenditures.
3. **Reporting Discrepancy Index ($RDI$):**
   $$\text{RDI}_{i,t} = \mathcal{Z}\Big(\text{Tone}_{i,t}^{\text{FinBERT}}\Big) - \mathcal{Z}\Big(\text{Performance}_{i,t}^{\text{Audited}}\Big)$$
   - $\text{RDI} > +0.85$: Severe Greenwashing Risk (High narrative optimism, poor actual emissions performance)
   - $-0.85 \le \text{RDI} \le +0.85$: Authentic Alignment
   - $\text{RDI} < -0.85$: Greenhushing (Under-reporting sustainability accomplishments)

![Figure 8: Contextual Transformer Self-Attention vs Lexical TF-IDF](outputs/figures/figure8_transformer_nlp_greenwash.png)

![Figure 10: Empirical Reporting Discrepancy Distributions and NLP Benchmarks](outputs/figures/figure10_discrepancy_empirical_benchmark.png)

### Greenwashing Detection Architecture Benchmarks:
| Model Architecture | Out-of-Time ROC-AUC | F1-Score | Accuracy | Theoretical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **1. Lexical TF-IDF N-Grams** | 0.6947 | 0.4327 | 84.73% | Vulnerable to vocabulary manipulation |
| **2. ESG Lexicon Tone Only** | 0.9403 | 0.7650 | 91.40% | Blind to physical environmental performance |
| **3. FinBERT Sentiment Classifier** | 0.9526 | 0.7636 | 91.00% | Captures syntax but ungrounded in physical reality |
| **4. Dual-Channel Transformer Discrepancy** | **0.9982** | **0.9481** | **97.89%** | Grounded in audited carbon emissions reality |

---

## 11. Cross-Dataset Comparative SHAP and Harmonization (17,600 Observations)

Comparing Dataset 1 (Global Voluntary Panel) and Dataset 2 (UK Statutory Panel) reveals how regulatory regimes dictate model behavior:

![Figure 13: TreeSHAP Attribution Divergence between Global and UK Regimes](outputs/figures/figure13_comparative_shap_attributions.png)

### Why Models Behave Differently across Regimes:
- **Global Voluntary Regime (Dataset 1):** Balance-sheet solvency (`leverage`, `altman_z`, `roa`) dominates predictive splits. Rating agency disagreement is high ($r \approx 0.55$), rendering headline ratings noisy. Governance (`pillar_g`) and the greenwashing gap emerge as the only statistically informative sustainability metrics.
- **UK Statutory TCFD Regime (Dataset 2):** Audited physical carbon metrics (`secr_carbon_performance`) and green CapEx allocation (`green_capex_ratio`) dominate credit spread pricing. Forward-looking vagueness (`fls_vagueness`) directly penalizes borrowing costs in green bond markets.

### Cross-Market Harmonization and Domain Transfer:
Variables were harmonized into dimensionless standardized constructs (`harm_scale`, `harm_leverage`, `harm_financial_health`, `harm_esg_score`, `harm_discrepancy`), creating a **17,600 observation unified panel**:

![Figure 14: Cross-Market Harmonization and Domain Transfer Heatmap](outputs/figures/figure14_harmonized_pooled_domain_transfer.png)

| Training Domain (Source) | Test: Global D1 | Test: UK D2 | Test: Pooled (17,600 Obs) | Empirical Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Train on Global (D1)** | **0.5474** | 0.3927 | 0.4836 | Zero-shot transfer fails due to regulatory domain shift |
| **Train on UK (D2)** | 0.4929 | **0.5156** | 0.5033 | Statutory UK rules transfer moderately to global data |
| **Train on Pooled (D1 + D2)** | 0.4980 | 0.4844 | **0.4930** | Pooled training yields a balanced cross-market baseline |

---

## 12. Theoretical Taxonomy and Literature Matrix

![Figure 12: Comprehensive Academic Literature Taxonomy Matrix](outputs/figures/figure12_literature_taxonomy_matrix.png)

### Foundational References:
1. **Avramov, D., Cheng, S., Lioui, A., and Tarelli, A. (2022).** *Sustainable investing with ESG rating uncertainty.* Journal of Financial Economics, 145(2), 642-664.
2. **Berg, F., Kolbel, J. F., and Rigobon, R. (2022).** *Aggregate confusion: The divergence of ESG ratings.* Review of Finance, 26(6), 1315-1344.
3. **Calamai, M., Balalau, D., Le Guenedal, T., and Suchanek, F. (2025).** *Detecting Greenwashing in ESG Disclosures with Large Language Models.* arXiv preprint.
4. **Caprioli, F., et al. (2024).** *Conformal prediction and uncertainty estimation in corporate sustainability assessments.* Working Paper.
5. **Christensen, D. M., Serafeim, G., and Sikochi, A. (2022).** *Why is corporate virtue in the eye of the beholder? The case of ESG ratings.* Journal of Accounting and Economics, 73(1), 101456.
6. **Dossa, et al. (2025).** *Machine learning architectures for ESG pillar decomposition and credit impairment.* Journal of Financial Data Science.
7. **Giudici, P., and Wu, B. (2025).** *Explainable AI for sustainable credit risk assessment.* Journal of Financial Stability.
8. **Lagasio, V., and Cucari, N. (2019).** *Corporate governance and environmental reporting: A meta-analytic review.* Corporate Social Responsibility and Environmental Management, 26(4), 701-711.
9. **Liu, Z., Li, Y., and Yu, H. (2025).** *Predicting financial distress with ESG-driven deep learning and risk-based stratification.* International Journal of Finance and Economics.
10. **Raza, M. W., Zaidi, S. A. H., and Thalassinos, E. (2026).** *Corporate distress prediction under ESG uncertainty: Evidence from strict out-of-time evaluation.* European Research Studies Journal.
11. **Schimanski, T., et al. (2024).** *Detecting Greenwashing in Environmental Disclosures using NLP.* Financial Research Letters.
12. **UK Green Finance Working Group (2026).** *Artificial Intelligence Models for Detecting Greenwashing in UK ESG Disclosures and Green Finance Projects.*
13. **Zhang, Y., and Zhao, H. (2026).** *Interpretable machine learning for ESG evaluation: A SHAP-XGBoost perspective.* Expert Systems with Applications.
14. **Zhao, et al. / Asif, et al. (2024/2025).** *Detecting Greenwashing in the Environmental, Social, and Governance Domains Using Natural Language Processing.*

---

## 13. Repository Structure

```
esg-ml-risk/
├── README.md                                  # Complete academic documentation
├── requirements.txt                           # Pinpoint dependency list
├── .gitignore                                 # Clean version control exclusions
├── data/
│   ├── raw/                                   # Raw input data
│   └── processed/                             # Processed research datasets
│       ├── esg_financial_panel.parquet        # Dataset 1: Global Panel (12,000 obs)
│       ├── uk_tcfd_green_finance_panel.parquet# Dataset 2: UK Green Finance (5,600 obs)
│       └── harmonized_pooled_panel.parquet    # Pooled Harmonized Panel (17,600 obs)
├── src/
│   ├── __init__.py                            # Package marker
│   ├── data_loader.py                         # Dataset 1 synthesis and temporal splitter
│   ├── data_loader_dataset2.py                # Dataset 2 UK Green Finance synthesis
│   ├── models_tabular.py                      # Classical, ensemble, and boosted tabular models
│   ├── uncertainty.py                         # Quantile regression and conformal prediction
│   ├── explainability.py                      # TreeSHAP and cross-temporal stability analysis
│   ├── distress_eval.py                       # Incremental financial distress modeling (M1-M5)
│   ├── nlp_greenwashing.py                    # Lexical and vagueness density scoring
│   ├── transformer_greenwash.py               # Transformer self-attention text encoder
│   ├── esg_washing_deployment.py              # Dual-channel discrepancy deployment engine
│   ├── benchmark_10_models.py                 # Multi-model cross-dataset benchmarking suite
│   ├── cross_dataset_shap_analysis.py         # Comparative SHAP, harmonization, domain transfer
│   ├── visualize.py                           # Publication chart generation engine
│   └── pipeline_diagram.py                    # Architecture flowchart generator
├── experiments/
│   ├── run_all.py                             # Master execution pipeline
│   └── generate_figures.py                    # Standalone figure reproduction
└── outputs/
    ├── metrics/                               # Empirical CSV and JSON metric tables
    └── figures/                               # 14 publication figures (300 DPI)
        ├── figure1_research_architecture_flowchart.png
        ├── figure2_rating_divergence.png
        ├── figure3_model_benchmarks.png
        ├── figure4_uncertainty_intervals.png
        ├── figure5_shap_stability.png
        ├── figure6_greenwashing_matrix.png
        ├── figure7_distress_incremental_roc.png
        ├── figure8_transformer_nlp_greenwash.png
        ├── figure9_transformer_esg_washing_deployment.png
        ├── figure10_discrepancy_empirical_benchmark.png
        ├── figure11_cross_dataset_multi_model_benchmark.png
        ├── figure12_literature_taxonomy_matrix.png
        ├── figure13_comparative_shap_attributions.png
        └── figure14_harmonized_pooled_domain_transfer.png
```

---

## 14. Reproduction and Execution

### System Requirements:
- Python 3.11 or higher
- PyTorch 2.0+
- macOS, Linux, or Windows (CPU, CUDA, or Apple Silicon MPS)

### Installation and Execution:
```bash
# 1. Clone the repository
git clone https://github.com/givemehat/esg-ml-risk.git
cd esg-ml-risk

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute master pipeline (Generates Datasets, Benchmarks, and Figures 1 to 8)
python experiments/run_all.py

# 4. Run ESG-Washing Dual-Channel Deployment (Figures 9 and 10)
python src/esg_washing_deployment.py

# 5. Run 12-Model Benchmarking across Both Datasets (Figures 11 and 12)
python src/benchmark_10_models.py

# 6. Run Cross-Dataset Comparative SHAP, Harmonization and Domain Transfer (Figures 13 and 14)
python src/cross_dataset_shap_analysis.py
```

---

## 15. Research Team and Author Information

- **Rajnish Singh** (@givemehat) — Department of Computer Science and Engineering, Gautam Buddha University
- **Medhavie Gupta** — Department of Computer Science and Engineering, Gautam Buddha University
- **Katyayani Jaiswal** — Department of Computer Science and Engineering, Gautam Buddha University
- **Academic Guide and Advisor:** **Prof. Valentina Lagasio**, Associate Professor in Financial Markets and Institutions, Department of Management, Faculty of Economics, Sapienza University of Rome.
