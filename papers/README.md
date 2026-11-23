# Research Literature & Reference Papers Library

This directory curates the empirical and theoretical research papers underlying our study on **Machine Learning for ESG Assessment, Rating Divergence, Financial Distress, and Greenwashing NLP**.

---

## Thematic Taxonomy & Research Mapping

```
papers/
├── Question 1: Divergence, Standardized Baselines & Risk Assessment
│   ├── Lagasio_2018_Corporate_Governance_Banks_MetaAnalysis.pdf
│   ├── Bank_of_England_Predicting_Bank_Distress_UK_Machine_Learning.pdf
│   ├── Credit_Growth_Yield_Curve_Financial_Crisis_Machine_Learning.pdf
│   ├── Mahmood_2024_ESG_Disclosure_Firm_Value_Europe.pdf
│   └── Aliamutu_2025_ESG_Disclosure_Firm_Profitability.pdf
│
└── Question 2: Transformer NLP & Greenwashing Discrepancy Detection
    ├── Lagasio_2024_ESG_Washing_Detection_IRFA.pdf (Primary Foundation)
    ├── Lagasio_2024_ESG_Washing_Summary_Deck.pdf
    ├── Lagasio_2024_ESG_Washing_Slides.pdf
    ├── Araci_2019_FinBERT_Financial_Sentiment.pdf
    ├── Webersinke_2021_ClimateBERT.pdf
    └── Lin_2024_FraudGT_Graph_Transformer_Financial_Fraud.pdf
```

---

## Detailed Annotated Bibliography

### 1. Primary Foundational Papers (ESG-Washing & NLP Discrepancies)

#### **Lagasio (2024)** — *ESG-washing detection in corporate sustainability reports*
- **File:** [`Lagasio_2024_ESG_Washing_Detection_IRFA.pdf`](file:///Users/rajnishsingh/Downloads/ESG_RESEARCH/papers/Lagasio_2024_ESG_Washing_Detection_IRFA.pdf)
- **Author:** Prof. Valentina Lagasio (*Department of Management, Sapienza University of Rome*)
- **Journal:** *International Review of Financial Analysis (IRFA)*
- **Key Contributions:**
  - Introduces the **ESG-washing Severity Index (ESGSI)** to quantitatively measure the discrepancy between portrayed ESG narratives and verified quantitative corporate performance.
  - Demonstrates that firms with lower actual environmental performance employ significantly higher positive disclosure tone and forward-looking linguistic obfuscation.
- **Role in our Study (Question 2):** Serves as the primary theoretical benchmark for our dual-channel FinBERT/ClimateBERT vs. quantitative KPI discrepancy formulation in `src/esg_washing_deployment.py` and `src/nlp_greenwashing.py`.

#### **Lagasio (2024) Summary Deck & Slides**
- **Files:** [`Lagasio_2024_ESG_Washing_Summary_Deck.pdf`](file:///Users/rajnishsingh/Downloads/ESG_RESEARCH/papers/Lagasio_2024_ESG_Washing_Summary_Deck.pdf), [`Lagasio_2024_ESG_Washing_Slides.pdf`](file:///Users/rajnishsingh/Downloads/ESG_RESEARCH/papers/Lagasio_2024_ESG_Washing_Slides.pdf)
- **Role in our Study:** High-level executive synthesis and visual presentation of the ESG-washing econometric specifications and policy implications.

---

### 2. Contextual Transformer Architectures & Financial NLP

#### **Araci (2019)** — *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*
- **File:** [`Araci_2019_FinBERT_Financial_Sentiment.pdf`](file:///Users/rajnishsingh/Downloads/ESG_RESEARCH/papers/Araci_2019_FinBERT_Financial_Sentiment.pdf)
- **Author:** Dogu Araci
- **Key Contributions:** Domain-specific pretraining of BERT on corporate financial communications (10-K, 10-Q, earnings calls), achieving substantial performance gains over dictionary-based sentiment methods (Loughran-McDonald, Harvard-IV).
- **Role in our Study (Question 2):** Serves as our primary neural feature extractor (`ProsusAI/finbert`) for corporate sustainability reports, capturing nuanced contextual sentiment polarity.

#### **Webersinke et al. (2021)** — *ClimateBERT: A Pretrained Language Model for Climate-Related Text*
- **File:** [`Webersinke_2021_ClimateBERT.pdf`](file:///Users/rajnishsingh/Downloads/ESG_RESEARCH/papers/Webersinke_2021_ClimateBERT.pdf)
- **Authors:** Nicolas Webersinke, Mathias Kraus, Julia Bingler, Markus Leippold
- **Key Contributions:** Adapts DistilRoBERTa specifically to climate change and ESG terminology, mitigating the high error rate of generic LLMs on technical environmental claims.
- **Role in our Study (Question 2):** Provides the specialized contextual backbone for extracting environmental claim specificity and detecting greenwashing commitments.

#### **Lin et al. (ACM 2024)** — *FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection*
- **File:** [`Lin_2024_FraudGT_Graph_Transformer_Financial_Fraud.pdf`](file:///Users/rajnishsingh/Downloads/ESG_RESEARCH/papers/Lin_2024_FraudGT_Graph_Transformer_Financial_Fraud.pdf)
- **Authors:** Junhong Lin, Xiaojie Guo, Yada Zhu, Samuel Mitchell, Erik Altman (MIT & IBM Research)
- **Conference:** *ACM KDD 2024*
- **Key Contributions:** Formulates graph-transformer attention over transaction networks and reporting relationships to identify fraudulent anomalies and reporting manipulations.
- **Role in our Study (Question 2):** Conceptual foundation for treating multi-channel reporting discrepancies as relational anomalies between text claims and audit nodes.

---

### 3. Rating Divergence, Corporate Governance & Financial Distress

#### **Lagasio (2018)** — *Corporate Governance in Banks: Systematic Literature Review and Meta-Analysis*
- **File:** [`Lagasio_2018_Corporate_Governance_Banks_MetaAnalysis.pdf`](file:///Users/rajnishsingh/Downloads/ESG_RESEARCH/papers/Lagasio_2018_Corporate_Governance_Banks_MetaAnalysis.pdf)
- **Author:** Prof. Valentina Lagasio
- **Journal:** *Corporate Ownership & Control*
- **Key Contributions:** Synthesizes the empirical impact of board composition, audit independence, and risk committee oversight on financial institutional resilience and risk mitigation.
- **Role in our Study (Question 1):** Provides the governance pillar baseline parameters for evaluating statutory oversight versus voluntary ratings.

#### **Mahmood, Mehmood, Terzani, & Pagani (2024)** — *The Effect of ESG Disclosure on Firm Value in the European Context*
- **File:** [`Mahmood_2024_ESG_Disclosure_Firm_Value_Europe.pdf`](file:///Users/rajnishsingh/Downloads/ESG_RESEARCH/papers/Mahmood_2024_ESG_Disclosure_Firm_Value_Europe.pdf)
- **Journal:** *Management Decision (2024)*
- **Key Contributions:** Examines non-linear valuation effects of ESG disclosures under European regulatory mandates (CSRD/NFRD), identifying market over-reaction and baseline disclosure effects.
- **Role in our Study (Question 1):** Evaluates how statutory disclosure standards reshape firm cost of capital and rating stability.

#### **Aliamutu et al. (2025/2026)** — *ESG Disclosure and Firm Profitability: Moderating Roles of Firm Size and Corporate Governance*
- **File:** [`Aliamutu_2025_ESG_Disclosure_Firm_Profitability.pdf`](file:///Users/rajnishsingh/Downloads/ESG_RESEARCH/papers/Aliamutu_2025_ESG_Disclosure_Firm_Profitability.pdf)
- **Journal:** *International Review of Management and Marketing*
- **Key Contributions:** Tests interaction terms between disclosure level, profitability (ROA/ROE), and enterprise scale across emerging and developed markets.
- **Role in our Study (Question 1):** Supports our control variable selection and multi-market panel harmonization.

#### **Bank of England Working Study** — *Predicting Bank Distress in the UK with Machine Learning*
- **File:** [`Bank_of_England_Predicting_Bank_Distress_UK_Machine_Learning.pdf`](file:///Users/rajnishsingh/Downloads/ESG_RESEARCH/papers/Bank_of_England_Predicting_Bank_Distress_UK_Machine_Learning.pdf)
- **Key Contributions:** Compares tree ensembles (XGBoost, Random Forest) against traditional supervisory CAMELS scoring for predicting insolvency under strict out-of-time temporal validation.
- **Role in our Study (Question 1):** Methodological benchmark for our strict temporal out-of-time evaluation (2018-2022 train vs. 2023-2025 test) in `src/distress_eval.py`.

#### **Empirical Macro-Finance Study** — *Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from Machine Learning*
- **File:** [`Credit_Growth_Yield_Curve_Financial_Crisis_Machine_Learning.pdf`](file:///Users/rajnishsingh/Downloads/ESG_RESEARCH/papers/Credit_Growth_Yield_Curve_Financial_Crisis_Machine_Learning.pdf)
- **Key Contributions:** Assesses machine learning generalization for tail-event financial crises and credit spread dislocations.
- **Role in our Study (Question 1):** Directly informs our credit spread prediction benchmark and out-of-time validation protocols.
