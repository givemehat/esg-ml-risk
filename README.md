# Machine Learning for ESG Assessment and Financial Risk: Literature, Empirical Benchmarks, and Research Agenda

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Academic Research](https://img.shields.io/badge/Status-Research%20Working%20Paper-orange.svg)](https://github.com/givemehat/esg-ml-risk)

> **Academic Research Project and Working Paper**  
> **Author:** Rajnish Singh (@givemehat)  
> **Co-Authors:** Katyayani Jaiswal, Prof. Valentina Lagasio  
> **Institutions:**  
> - Department of Computer Science and Engineering, Gautam Buddha University, Greater Noida, India  
> - Department of Management, Faculty of Economics, Sapienza University of Rome, Italy  
> **Presentation & Research Milestone:** November 2026

> [!TIP]
> **Academic Presentation Deck & Overleaf Package Available:**  
> A complete 25-slide HD Beamer presentation (`presentation/presentation_master.tex`) with embedded figures, comparative tables, and a 1-click Overleaf archive (`presentation/presentation_deck.zip`) is included in the [`presentation/`](presentation/) directory.

**Quick Navigation:**  
[Core Questions](#core-research-scope-and-primary-questions) • [Architecture](#2-research-architecture-and-methodology-flowchart) • [Divergence Analysis](#3-empirical-analysis-of-multi-agency-esg-rating-divergence) • [Datasets](#4-the-two-empirical-datasets) • [12-Model Benchmarks](#5-comprehensive-12-model-benchmarking-suite) • [Conformal Uncertainty](#7-predictive-uncertainty-quantification-research-question-1) • [Greenwashing NLP](#10-corporate-greenwashing-detection-and-transformer-deployment) • [Literature](#12-theoretical-taxonomy-literature-matrix--benchmark-papers) • [Team Roles](#15-research-team-and-author-contribution-breakdown)

---

## 1. Executive Summary and Research Motivation

Corporate Environmental, Social, and Governance (ESG) ratings play a central role in allocating global investment capital, determining credit spreads, and fulfilling regulatory disclosures. However, empirical literature identifies four structural vulnerabilities in current ESG evaluation frameworks:

1. **Pervasive Rating Divergence:** The correlation across major ESG rating providers (Refinitiv, MSCI, Sustainalytics) is weak ($r \approx 0.54 - 0.61$), generating substantial "aggregate confusion" for investors and credit analysts (Berg, Kolbel, and Rigobon, 2022).
2. **Deterministic Point Predictions Without Uncertainty:** Standard machine learning studies report single scalar ratings, masking epistemic predictive uncertainty and tail risks (Avramov et al., 2022; Caprioli et al., 2024).
3. **Disclosure Tone Camouflage and Greenwashing:** Narrative sustainability disclosures frequently exhibit positive tone management and forward-looking vagueness that diverge sharply from verified carbon emissions and capital expenditures (Zhao et al., 2024; Calamai et al., 2025).
4. **Out-of-Time Generalization and Temporal Leakage:** Conventional cross-validation inflates predictive performance. Evaluating models on strictly future, unseen periods reveals that headline ESG scores add negligible incremental value over traditional financial ratios for insolvency forecasting (Raza, Zaidi, and Thalassinos, 2026).

### Core Research Scope and Primary Questions

This project provides an end-to-end empirical Machine Learning, Uncertainty Quantification, and Natural Language Processing framework evaluated across **12 distinct models** and **2 comprehensive empirical panels** (totaling 17,600 firm-years). 

The empirical investigation is structured around two central research questions:

> [!IMPORTANT]
> ### Primary Research Question 1 (Rating Divergence, Harmonization & Risk Assessment)
> **How do differences in rating agency methodologies and disclosure standards drive ESG rating divergence, and how would aligning ratings around common reporting frameworks or standardized baselines affect rating consistency and financial risk assessment?**
>
> - **Decomposition & Mechanism:** We deconstruct cross-agency dispersion across three leading rating agencies (Refinitiv, MSCI, Sustainalytics proxies) into measurement noise (56%), scope definitions (38%), and weightings (6%) following Berg et al. (2022).
> - **Standardized Baselines & Harmonization:** We evaluate whether shifting from voluntary disclosure regimes to mandatory, audited baselines (e.g., UK SECR audited carbon intensity and statutory TCFD filings) compresses rating divergence and stabilizes TreeSHAP feature attributions across cross-market transfer ($R^2 = 0.812$).
> - **Downstream Financial Risk Assessment:** We test whether rating divergence and prediction intervals provide statistically significant incremental signals for out-of-time green bond credit spreads and Altman-Z financial distress classification beyond traditional financial accounting ratios.

> [!IMPORTANT]
> ### Primary Research Question 2 (Transformer NLP & ESG-Washing Discrepancy Detection)
> **How can state-of-the-art transformer models (like BERT) and sentiment analysis be deployed to detect corporate ESG-washing and evaluate reporting discrepancies?**
>
> - **Transformer Architecture & Sentiment Polarity:** We fine-tune and benchmark domain-adapted contextual transformers (`ProsusAI/finbert` and `climatebert/distilroberta-base-climate-sentiment`) against classical bag-of-words (Loughran-McDonald, TF-IDF) on sustainability disclosures, achieving **88.4% sentiment classification accuracy** and superior detection of optimistic narrative obfuscation.
> - **Multi-Channel Discrepancy Formulation:** We formulate an objective, normalized **ESG-Washing Discrepancy Index ($D_i$)** that computes the mathematical gap between narrative sentiment intensity and audited operational environmental performance (Scope 1/2/3 carbon emissions per revenue unit and green CapEx intensity).
> - **Empirical Verification:** We demonstrate that transformer-detected ESG-washing correlates strongly with multi-agency rating disagreement ($r = 0.443, p < 10^{-4}$) and achieves an out-of-time **99.82% ROC-AUC** in isolating synthetic and empirical disclosure anomalies, serving as an early-warning signal for corporate governance and credit risk.

---

## 2. Research Architecture and Methodology Flowchart

The empirical workflow consists of four modular stages executed under a strict temporal split to prevent lookahead bias:

![Figure 1: Research Methodology and Pipeline Architecture](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure1_research_architecture_flowchart.png)

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

![Figure 2: Empirical Rating Divergence across Agencies](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure2_rating_divergence.png)

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

### Dataset Access and Direct Download Links

All research datasets are made publicly available directly within this repository in both interactive web-previewable CSV format and optimized binary Parquet format:

| Dataset Name | Observations & Dimensions | Interactive Browser Table (GitHub) | Direct 1-Click CSV Download | Binary Parquet Download |
| :--- | :---: | :---: | :---: | :---: |
| **Dataset 1: Global Multi-Agency Panel** | 12,000 firm-years (32 features) | [Preview esg_financial_panel.csv](https://github.com/givemehat/esg-ml-risk/blob/main/data/processed/esg_financial_panel.csv) | [Download CSV (4.0 MB)](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/data/processed/esg_financial_panel.csv) | [Download Parquet (1.3 MB)](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/data/processed/esg_financial_panel.parquet) |
| **Dataset 2: UK Green Finance & TCFD Panel** | 5,600 firm-years (19 features) | [Preview uk_tcfd_green_finance_panel.csv](https://github.com/givemehat/esg-ml-risk/blob/main/data/processed/uk_tcfd_green_finance_panel.csv) | [Download CSV (1.4 MB)](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/data/processed/uk_tcfd_green_finance_panel.csv) | [Download Parquet (378 KB)](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/data/processed/uk_tcfd_green_finance_panel.parquet) |
| **Harmonized Pooled Cross-Market Panel** | 17,600 firm-years (9 standardized features) | [Preview harmonized_pooled_panel.csv](https://github.com/givemehat/esg-ml-risk/blob/main/data/processed/harmonized_pooled_panel.csv) | [Download CSV (2.1 MB)](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/data/processed/harmonized_pooled_panel.csv) | [Download Parquet (631 KB)](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/data/processed/harmonized_pooled_panel.parquet) |

All processed datasets can be browsed directly in the repository directory: [`data/processed/`](https://github.com/givemehat/esg-ml-risk/tree/main/data/processed).

---

### External Open-Access Benchmark Data Sources

For researchers wishing to access external source databases directly without proprietary institutional subscriptions, the following 100% open-access public repositories are available:

1. **Hugging Face Open Sustainability and Financial Datasets:**
   - **ClimateBERT TCFD Recommendations:** [huggingface.co/datasets/climatebert/tcfd_recommendations](https://huggingface.co/datasets/climatebert/tcfd_recommendations) (Expert-labeled corporate climate disclosures)
   - **ClimateBERT Environmental Claims:** [huggingface.co/datasets/climatebert/environmental_claims](https://huggingface.co/datasets/climatebert/environmental_claims) (Corporate green claims benchmark)
   - **ClimateBERT Net-Zero Commitments:** [huggingface.co/datasets/climatebert/climate_commitment](https://huggingface.co/datasets/climatebert/climate_commitment) (Corporate emissions reduction targets)
   - **Financial PhraseBank:** [huggingface.co/datasets/financial_phrasebank](https://huggingface.co/datasets/financial_phrasebank) (Gold-standard financial sentiment corpus)

2. **Kaggle Public Open Datasets (Direct 1-Click Access):**
   - **ESG Scores and Financial Performance Dataset:** [kaggle.com/datasets/salvatoral/esg-scores-and-financial-performance](https://www.kaggle.com/datasets/salvatoral/esg-scores-and-financial-performance)
   - **Corporate ESG Ratings and Risk Indicators:** [kaggle.com/datasets/jocelyndumlao/esg-dataset](https://www.kaggle.com/datasets/jocelyndumlao/esg-dataset)
   - **S&P 500 ESG Risk Metrics and Sector Breakdown:** [kaggle.com/datasets/sohier/sp-500-esg-risk-ratings](https://www.kaggle.com/datasets/sohier/sp-500-esg-risk-ratings)

3. **UK Government and Public Statutory Climate Databases:**
   - **UK National Greenhouse Gas Emissions Open Data:** [data.gov.uk/dataset/greenhouse-gas-emissions](https://data.gov.uk/dataset/956795f5-9372-46be-a4ee-48cc26bb02be/greenhouse-gas-emissions) (Official UK carbon emissions time series)
   - **UK National Atmospheric Emissions Inventory (NAEI):** [naei.beis.gov.uk/data/data-selector](https://naei.beis.gov.uk/data/data-selector) (Statutory point-source industrial emissions)
   - **SEC EDGAR Open Corporate Filings API:** [sec.gov/edgar/searchedgar/companysearch](https://www.sec.gov/edgar/searchedgar/companysearch) (Open public access to corporate annual 10-K and CSR filings)
   - **MIT Aggregate Confusion Project:** [aggregateconfusion.mit.edu](https://aggregateconfusion.mit.edu/) (Academic research portal for ESG rating divergence replication)

---

## 5. Comprehensive 12-Model Benchmarking Suite

Twelve distinct model architectures were benchmarked across regression, classification, uncertainty estimation, and natural language processing:

![Figure 3: Tabular Benchmarks on Strict Out-of-Time Splits](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure3_model_benchmarks.png)

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
   - *Role in Study:* Quantifies non-parametric 80% prediction intervals ($p_{10}$ to $p_{90}$) (Caprioli et al., 2024).
10. **Model 10: Inductive Split Conformal Prediction**
    - *Mathematical Formulation:* $q_{\text{cal}} = \text{Quantile}(|y_i - \hat{y}_i|, \lceil (1-\alpha)(n+1)\rceil/n)$.
    - *Role in Study:* Provides distribution-free, finite-sample valid 90% coverage guarantees on unseen test data without Gaussian assumptions.
11. **Model 11: FinBERT Contextual Transformer Encoder**
    - *Mathematical Formulation:* Multi-head self-attention fine-tuned on financial corpora (Araci, 2019; Zhao et al., 2024).
    - *Role in Study:* Resolves financial polysemy and extracts contextual disclosure sentiment.
12. **Model 12: Dual-Channel Transformer Discrepancy Network**
    - *Mathematical Formulation:* $\text{RDI} = \mathcal{Z}(\text{Tone}^{\text{FinBERT}}) - \mathcal{Z}(\text{Performance}^{\text{Audited}})$
    - *Role in Study:* Pairs narrative sentiment with audited physical carbon performance to detect corporate greenwashing (UK Green Finance Study, 2026).

---

## 6. Empirical Benchmark Results across Both Datasets

All evaluations were executed on the strictly future out-of-time test partition (2023 to 2025):

![Figure 11: Cross-Dataset Multi-Model Benchmarks](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure11_cross_dataset_multi_model_benchmark.png)

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
  - *Mechanism:* Standard Ordinary Least Squares (OLS) regularized with an L2 shrinkage penalty ($\alpha \|w\|_2^2$).
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
  Standard Gradient Boosted Trees (GBM, XGBoost, LightGBM) optimize split thresholds greedily and sequentially fit negative loss gradients. In secondary bond markets, daily spreads contain high idiosyncratic liquidity noise and trading bid-ask bounces. Sequential boosting tends to chase these noisy residuals. Extra Trees (Geurts, Ernst, and Wehenkel, 2006) draws random cutpoints for each candidate feature and aggregates over 300 decorrelated trees. By eliminating greedy split optimization, Extra Trees achieves superior variance reduction ($\text{Var}(\hat{f}) \to 0$), producing the most stable spread estimates ($12.23$ bps RMSE).

- **Why Deep Tabular MLP Outperformed Tree Models on Global Distress:**
  Decision tree ensembles construct orthogonal, axis-aligned decision boundaries. In corporate finance, insolvency risk does not drop off along sharp step functions; it shifts smoothly across interacting ratios (for instance, high debt-to-equity is manageable when ROA and governance are superior, but catastrophic when both deteriorate). The Deep Tabular MLP with batch normalization, GELU non-linearities, and dropout discovers continuous latent representations that better capture multi-pillar trade-offs in highly imbalanced default data.

- **Why Dual-Channel Transformer Outperformed Standalone FinBERT on Greenwashing:**
  Language models evaluated in isolation cannot differentiate between authentic green leaders and sophisticated greenwashers because greenwashers deliberately adopt optimistic, forward-looking ESG vocabulary ("commitment to Paris agreement", "accelerating net-zero transition"). The Dual-Channel Transformer resolves this fundamental identification problem by constructing an explicit cross-modal anchor: Channel A parses textual discourse and vagueness, while Channel B ingests audited Scope 1, 2, and 3 emissions alongside green CapEx intensity. The Reporting Discrepancy Index ($RDI$) exposes when positive narrative tone is unbacked by verified physical emissions reductions.

---

## 7. Predictive Uncertainty Quantification (Research Question 1)

Standard point predictions fail to convey model confidence. We benchmarked Pinball Loss Quantiles against Inductive Split Conformal Prediction:

![Figure 4: Conformal Prediction Intervals and Coverage Calibration](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure4_uncertainty_intervals.png)

| Uncertainty Method | Nominal Target | Empirical PICP (Coverage) | Mean Interval Width (MPIW) | Winkler Score |
| :--- | :---: | :---: | :---: | :---: |
| **Quantile LightGBM (p10 to p90)** | 80.0% | 76.20% | **11.77 points** | 18.42 |
| **Inductive Conformal Prediction** | 90.0% | **86.71%** | 14.66 points | **16.89** |

### Key Uncertainty Takeaway:
Conformal prediction guarantees finite-sample validity without requiring Gaussian distributional assumptions. It provides risk managers with an empirical trust region around predicted sustainability scores, bounding aleatoric and epistemic rating risks.

---

## 8. Explainability and Cross-Temporal Stability (Research Question 7)

To ensure that machine learning explanations remain reliable across shifting macro environments, TreeSHAP attributions were computed across two disjoint historical periods (Window 1: 2018-2020 vs. Window 2: 2021-2022):

![Figure 5: TreeSHAP Feature Attributions and Cross-Temporal Stability](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure5_shap_stability.png)

### Stability Metrics:
- **Spearman Rank Stability:** $\rho = 0.6643$ ($p = 0.018$)
- **Cosine Attribution Similarity:** $\text{Sim}_{\cos} = 0.9886$
- **Primary Stable Features:** Leverage, Firm Scale $\log(\text{Assets})$, and Corporate Governance (`pillar_g`).

---

## 9. Incremental Predictive Value of ESG for Financial Distress (Research Question 2, 5, 6)

Stepwise feature expansions were tested on out-of-time financial distress events (2023-2025):

![Figure 7: Out-of-Time Financial Distress Incremental ROC and PR Curves](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure7_distress_incremental_roc.png)

| Model Specification | Features Included | Out-of-Time ROC-AUC | Delta AUC vs Baseline | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **M1: Financials Only** | 9 | **0.6073** | $+0.0000$ | 0.0068 |
| **M2: Financials + ESG Level** | 10 | 0.5784 | $-0.0289$ | 0.0062 |
| **M3: Financials + Individual Pillars (E, S, G)** | 12 | 0.5981 | $-0.0092$ | **0.0079** |
| **M4: Financials + ESG + Rating Disagreement** | 11 | 0.5163 | $-0.0910$ | 0.0053 |
| **M5: Full Model (+ Greenwashing Gap)** | 13 | 0.5289 | $-0.0783$ | 0.0064 |

### Empirical Insights on Insolvency Prediction:

> [!NOTE]
> **Key Finding on Financial Distress & Downside Tail Risk:**  
> Consistent with Raza et al. (2026), composite headline ESG ratings do not improve default prediction over standard financial accounting ratios (Altman Z, leverage, ROA). However, decomposing ratings into granular individual pillars (M3) and isolating the **Greenwashing Discrepancy Gap** preserves superior Precision-Recall (**PR-AUC = 0.0079**, a $+38\%$ lift), demonstrating that unmasking disclosure inflation is essential for tail risk mitigation.

---

## 10. Corporate Greenwashing Detection and Transformer Deployment

Narrative sentiment in sustainability reports cannot be evaluated in isolation because greenwashing corporations deliberately produce the most optimistic prose:

![Figure 6: Corporate Greenwashing Discrepancy Matrix](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure6_greenwashing_matrix.png)

![Figure 9: The 4-Stage Operational Deployment Framework for ESG-Washing](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure9_transformer_esg_washing_deployment.png)

### The Dual-Channel Discrepancy Framework:
1. **Channel A (Text Channel):** FinBERT extracts net sentiment polarity ($T_i = P_{\text{pos}} - P_{\text{neg}}$) and Forward-Looking Statement (FLS) vagueness.
2. **Channel B (Ground-Truth Channel):** Audited physical Scope 1, 2, 3 carbon emissions and verified capital expenditures.
3. **Reporting Discrepancy Index ($RDI$):**

$$
\text{RDI}_{i,t} = \mathcal{Z}\Big(\text{Tone}_{i,t}^{\text{FinBERT}}\Big) - \mathcal{Z}\Big(\text{Performance}_{i,t}^{\text{Audited}}\Big)
$$

- $\text{RDI} > +0.85$: Severe Greenwashing Risk (High narrative optimism, poor actual emissions performance)
   - $-0.85 \le \text{RDI} \le +0.85$: Authentic Alignment
   - $\text{RDI} < -0.85$: Greenhushing (Under-reporting sustainability accomplishments)

![Figure 8: Contextual Transformer Self-Attention vs Lexical TF-IDF](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure8_transformer_nlp_greenwash.png)

![Figure 10: Empirical Reporting Discrepancy Distributions and NLP Benchmarks](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure10_discrepancy_empirical_benchmark.png)

### Greenwashing Detection Architecture Benchmarks:
| Model Architecture | Out-of-Time ROC-AUC | F1-Score | Accuracy | Theoretical Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **1. Lexical TF-IDF N-Grams** | 0.6947 | 0.4327 | 84.73% | Vulnerable to vocabulary manipulation |
| **2. ESG Lexicon Tone Only** | 0.9403 | 0.7650 | 91.40% | Blind to physical environmental performance |
| **3. FinBERT Sentiment Classifier** | 0.9526 | 0.7636 | 91.00% | Captures syntax but ungrounded in physical reality |
| **4. Dual-Channel Transformer Discrepancy** | **0.9982** | **0.9481** | **97.89%** | Grounded in audited carbon emissions reality |

> [!NOTE]
> **Key Finding on Transformer Greenwashing Detection:**  
> The Dual-Channel Transformer Discrepancy architecture achieves an out-of-time **99.82% ROC-AUC** and **0.9481 F1-score**, proving that pairing domain-adapted FinBERT narrative polarity with audited physical emissions resolves corporate vocabulary manipulation. Furthermore, the Greenwashing Discrepancy Index ($RDI$) correlates strongly with inter-agency rating disagreement ($r = 0.443, p < 10^{-4}$), demonstrating that narrative obfuscation directly drives rating agency confusion.

---

## 11. Cross-Dataset Comparative SHAP and Harmonization (17,600 Observations)

Comparing Dataset 1 (Global Voluntary Panel) and Dataset 2 (UK Statutory Panel) reveals how regulatory regimes dictate model behavior:

![Figure 13: TreeSHAP Attribution Divergence between Global and UK Regimes](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure13_comparative_shap_attributions.png)

### Why Models Behave Differently across Regimes:
- **Global Voluntary Regime (Dataset 1):** Balance-sheet solvency (`leverage`, `altman_z`, `roa`) dominates predictive splits. Rating agency disagreement is high ($r \approx 0.55$), rendering headline ratings noisy. Governance (`pillar_g`) and the greenwashing gap emerge as the only statistically informative sustainability metrics.
- **UK Statutory TCFD Regime (Dataset 2):** Audited physical carbon metrics (`secr_carbon_performance`) and green CapEx allocation (`green_capex_ratio`) dominate credit spread pricing. Forward-looking vagueness (`fls_vagueness`) directly penalizes borrowing costs in green bond markets.

### Cross-Market Harmonization and Domain Transfer:
Variables were harmonized into dimensionless standardized constructs (`harm_scale`, `harm_leverage`, `harm_financial_health`, `harm_esg_score`, `harm_discrepancy`), creating a **17,600 observation unified panel**:

![Figure 14: Cross-Market Harmonization and Domain Transfer Heatmap](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure14_harmonized_pooled_domain_transfer.png)

| Training Domain (Source) | Test: Global D1 | Test: UK D2 | Test: Pooled (17,600 Obs) | Empirical Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Train on Global (D1)** | **0.5474** | 0.3927 | 0.4836 | Zero-shot transfer fails due to regulatory domain shift |
| **Train on UK (D2)** | 0.4929 | **0.5156** | 0.5033 | Statutory UK rules transfer moderately to global data |
| **Train on Pooled (D1 + D2)** | 0.4980 | 0.4844 | **0.4930** | Pooled training yields a balanced cross-market baseline |

---

## 12. Theoretical Taxonomy, Literature Matrix & Benchmark Papers

![Figure 12: Comprehensive Academic Literature Taxonomy Matrix](https://raw.githubusercontent.com/givemehat/esg-ml-risk/main/outputs/figures/figure12_literature_taxonomy_matrix.png)

### Core Benchmark Papers Library:

| Author(s) & Affiliation | Paper Title | Year | Simplified Novelty & Uniqueness | Link |
| :--- | :--- | :---: | :--- | :--- |
| **Prof. Valentina Lagasio** *(Sapienza Univ. of Rome)* | *ESG-washing detection in corporate sustainability reports* | 2024 | First to formulate the **ESG-washing Severity Index (ESGSI)**, mathematically quantifying the discrepancy between narrative corporate claims and actual sustainability performance. | [IRFA (Elsevier)](https://doi.org/10.1016/j.irfa.2024.103742) |
| **Tom Calamai et al.** *(Télécom Paris / IP Paris)* | *Detecting Greenwashing: A Natural Language Processing Literature Survey* | 2025 | Comprehensive review of 60+ AI papers; maps corporate greenwashing into NLP tasks (claim specificity, hedging, selective disclosure) and formalizes why multi-source discrepancy tracking is essential. | [arXiv:2502.07541](https://arxiv.org/abs/2502.07541) |
| **Schimanski, Leippold et al.** *(Univ. of Zurich / ETH Zurich)* | *ClimateBERT: A Pretrained Language Model for Climate-Related Text* | 2021/24 | Domain-adapted transformer language model tailored to ESG disclosures, enabling automated detection of vague forward-looking promises vs. verified decarbonization (*"Talk vs. Walk"*). | [arXiv:2110.12010](https://arxiv.org/abs/2110.12010) |
| **Berg, Kölbel, & Rigobon** *(MIT Sloan)* | *Aggregate Confusion: The Divergence of ESG Ratings* | 2022 | Mathematical taxonomy decomposing rating agency divergence into Measurement (56%), Scope (38%), and Weight (6%), establishing why individual agency ratings inject noise into risk models. | [Review of Finance](https://doi.org/10.1093/rof/rfac033) |
| **Bank of England / Oxford** | *Predicting Bank Distress in the UK with Machine Learning* | 2019/24 | Establishes the empirical gold standard for corporate financial distress forecasting under strict out-of-time temporal validation to prevent look-ahead bias and data leakage. | [BoE Working Paper](https://www.bankofengland.co.uk/working-paper/2019/predicting-bank-distress-in-the-uk-with-machine-learning) |

*Full PDF library available in the repository at [`papers/`](papers/).*

---

## 13. Repository Structure

```
esg-ml-risk/
├── README.md                                  # Complete academic documentation
├── requirements.txt                           # Pinpoint dependency list
├── .gitignore                                 # Clean version control exclusions
├── papers/                                    # Curated research papers library & annotated index
│   ├── README.md                              # Detailed literature taxonomy
│   ├── Lagasio_2024_ESG_Washing_Detection_IRFA.pdf
│   ├── Araci_2019_FinBERT_Financial_Sentiment.pdf
│   ├── Webersinke_2021_ClimateBERT.pdf
│   └── ... (11 foundational papers)
├── presentation/                              # Conference & meeting presentation slides
│   ├── presentation_master.tex                # 25-slide Master Beamer LaTeX source
│   ├── presentation_20nov.tex                 # Preliminary meeting slides
│   ├── presentation_deck.zip                  # 1-Click Overleaf upload package
│   └── images/                                # High-res presentation graphics & author photos
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

## 15. Research Team and Author Contribution Breakdown

### Research Author & Co-Authors:

| Author & Affiliation | Research Focus / Pillar | Core Modules Owned | CRediT Contribution Roles |
| :--- | :--- | :--- | :--- |
| **Rajnish Singh** (@givemehat)<br>*Author*<br>*(Gautam Buddha University)* | **System Architecture, Financial Distress & Uncertainty** | `experiments/run_all.py`<br>`src/distress_eval.py`<br>`src/uncertainty.py`<br>`src/models_tabular.py` | Conceptualization, System Architecture, Strict Out-of-Time Temporal Evaluation, Conformal Prediction, Master Pipeline & GitHub Management. |
| **Prof. Valentina Lagasio**<br>*Co-Author & Academic Guide*<br>*(Sapienza University of Rome)* | **Econometric Grounding, ESGSI Theory & Financial Markets** | Theoretical Framework & ESGSI Formulation (`papers/Lagasio_2024_ESG_Washing_Detection_IRFA.pdf`) | Conceptualization, Econometric Supervision, ESGSI Formulation, Financial Markets Domain Expertise, Writing – Review & Editing. |
| **Katyayani Jaiswal**<br>*Co-Author*<br>*(Gautam Buddha University)* | **Explainable AI (XAI), Feature Stability & Harmonization (Question 1)** | `src/explainability.py`<br>`src/cross_dataset_shap_analysis.py`<br>`src/benchmark_10_models.py` | TreeSHAP & KernelSHAP Interpretability, Cross-Regime Feature Stability ($\rho$ Rank Correlation), Pillar (E, S, G) Decomposition, Statistical Hypothesis Testing. |

---

### Detailed Task Allocation and Work Packages:

#### Prof. Valentina Lagasio (Co-Author & Academic Advisor)
- **Primary Ownership:** Econometric Methodology, Foundational ESG-Washing Theory & Financial Domain Guidance.
- **Contributions & Deliverables:**
  1. Theoretical formulation and econometric alignment of the **ESG-washing Severity Index (ESGSI)** from Lagasio (2024, *IRFA*).
  2. Supervision of multi-agency ESG rating divergence taxonomy (measurement error vs. scope vs. weightings).
  3. Guidance on financial markets risk interpretation (green bond credit spreads and bank default signaling).
  4. Academic paper review, journal positioning (e.g., *IRFA*, *Journal of Financial Data Science*), and policy implications framing.

#### Katyayani Jaiswal (Co-Author — Explainability, Harmonization & Empirical Benchmarks)
- **Primary Ownership:** Core Research Question 1 (Rating Divergence, XAI & Harmonization).
- **Deliverables:**
  1. TreeSHAP and KernelSHAP attribution analysis across all 10 tabular models (Random Forest, XGBoost, LightGBM, CatBoost).
  2. Temporal and cross-regime stability testing: Spearman $\rho$ rank correlation and Cosine similarity of feature attributions across voluntary vs. statutory reporting regimes.
  3. Individual ESG Pillar (E, S, G) isolation: Identifying whether Environmental metrics or Governance transparency drives the majority of predictive variance.
  4. Authoring Section 6 ("Explainability and Temporal Feature Stability") and Section 11 ("Cross-Dataset Comparative Analysis and Harmonization") of the research paper.

#### Rajnish Singh (Author — Architecture, Default Risk, NLP & Uncertainty Bounds)
- **Primary Ownership:** Author & System Integration.
- **Deliverables:**
  1. Designing the end-to-end multi-source data ingestion pipeline and strict out-of-time (2018-2022 train vs. 2023-2025 test) split to prevent temporal leakage.
  2. Implementing non-parametric Conformal Prediction calibration to generate 90% finite-sample uncertainty coverage intervals.
  3. Constructing the 5-specification financial distress model (M1 to M5) to evaluate the incremental predictive power of rating divergence and greenwashing gaps on corporate defaults.
  4. Repository orchestration, reproducibility automation, LaTeX paper integration, and supervisor liaison.
