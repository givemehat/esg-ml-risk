"""
Transformer / BERT Greenwashing Representation Module
Implements a multi-head self-attention Transformer text encoder in PyTorch.
Extracts contextual embeddings from corporate CSR disclosures and benchmarks against lexical/TF-IDF baselines.
Based on Calamai et al. (2025) and Schimanski et al. (2024).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(name)s: %(message)s')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class ESGTransformerEncoder(nn.Module):
    """
    Multi-head Self-Attention Transformer Architecture for Corporate ESG Disclosure Analysis.
    """
    def __init__(self, vocab_size: int = 2000, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.fc_pool = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch_size, seq_len)
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        out = self.transformer_encoder(emb, src_key_padding_mask=mask)
        # Global average pooling over non-padding tokens
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            pooled = torch.sum(out * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        else:
            pooled = torch.mean(out, dim=1)
        return torch.tanh(self.fc_pool(pooled))

class SimpleTokenizer:
    def __init__(self, max_vocab: int = 2000):
        self.max_vocab = max_vocab
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, texts: List[str]):
        import collections
        counter = collections.Counter()
        for t in texts:
            words = t.lower().split()
            counter.update(words)
        for w, _ in counter.most_common(self.max_vocab - 2):
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w

    def encode(self, texts: List[str], max_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = []
        masks = []
        for t in texts:
            words = t.lower().split()
            tokens = [self.word2idx.get(w, 1) for w in words[:max_len]]
            pad_len = max_len - len(tokens)
            pad_tokens = tokens + [0] * pad_len
            mask = [False] * len(tokens) + [True] * pad_len
            encoded.append(pad_tokens)
            masks.append(mask)
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

def compare_transformer_vs_tfidf(
    texts: List[str],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
    """
    Benchmarks Transformer Self-Attention representations against TF-IDF n-grams for greenwashing detection.
    """
    # 1. TF-IDF Baseline
    tfidf = TfidfVectorizer(max_features=256, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform([texts[i] for i in train_idx])
    X_test_tfidf = tfidf.transform([texts[i] for i in test_idx])

    clf_tfidf = LogisticRegression(max_iter=500, random_state=42)
    clf_tfidf.fit(X_train_tfidf, labels[train_idx])
    probs_tfidf = clf_tfidf.predict_proba(X_test_tfidf)[:, 1]
    preds_tfidf = (probs_tfidf >= 0.5).astype(int)

    metrics_tfidf = {
        "ROC-AUC": round(float(roc_auc_score(labels[test_idx], probs_tfidf)), 4),
        "F1-Score": round(float(f1_score(labels[test_idx], preds_tfidf, zero_division=0)), 4),
        "Accuracy": round(float(accuracy_score(labels[test_idx], preds_tfidf)), 4)
    }

    # 2. Transformer Contextual Embeddings
    logger.info("Fitting tokenizer and running Transformer Multi-Head Self-Attention...")
    tokenizer = SimpleTokenizer(max_vocab=2000)
    tokenizer.fit(texts)
    input_ids, masks = tokenizer.encode(texts, max_len=64)

    torch.manual_seed(42)
    device = "cpu"
    model = ESGTransformerEncoder(vocab_size=2000, d_model=64, nhead=4, num_layers=2).to(device)
    model.eval()

    with torch.no_grad():
        all_embeddings = []
        batch_size = 256
        for i in range(0, len(texts), batch_size):
            b_ids = input_ids[i:i + batch_size].to(device)
            b_masks = masks[i:i + batch_size].to(device)
            emb = model(b_ids, b_masks)
            all_embeddings.append(emb.cpu().numpy())
        embeddings = np.vstack(all_embeddings)

    X_train_tf = embeddings[train_idx]
    X_test_tf = embeddings[test_idx]

    clf_tf = LogisticRegression(max_iter=500, random_state=42)
    clf_tf.fit(X_train_tf, labels[train_idx])
    probs_tf = clf_tf.predict_proba(X_test_tf)[:, 1]
    preds_tf = (probs_tf >= 0.5).astype(int)

    metrics_transformer = {
        "ROC-AUC": round(float(roc_auc_score(labels[test_idx], probs_tf)), 4),
        "F1-Score": round(float(f1_score(labels[test_idx], preds_tf, zero_division=0)), 4),
        "Accuracy": round(float(accuracy_score(labels[test_idx], preds_tf)), 4)
    }

    return metrics_tfidf, metrics_transformer, embeddings

def plot_transformer_comparison(
    metrics_tfidf: dict,
    metrics_transformer: dict,
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str
):
    """
    Figure 8: Transformer vs TF-IDF Comparison and Contextual Embedding Semantic Space
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Panel A: Performance Comparison Barplot
    ax1 = axes[0]
    df_comp = pd.DataFrame([
        {"Method": "TF-IDF + Logistic", "Metric": "ROC-AUC", "Score": metrics_tfidf["ROC-AUC"]},
        {"Method": "Transformer (Multi-Head)", "Metric": "ROC-AUC", "Score": metrics_transformer["ROC-AUC"]},
        {"Method": "TF-IDF + Logistic", "Metric": "F1-Score", "Score": metrics_tfidf["F1-Score"]},
        {"Method": "Transformer (Multi-Head)", "Metric": "F1-Score", "Score": metrics_transformer["F1-Score"]},
        {"Method": "TF-IDF + Logistic", "Metric": "Accuracy", "Score": metrics_tfidf["Accuracy"]},
        {"Method": "Transformer (Multi-Head)", "Metric": "Accuracy", "Score": metrics_transformer["Accuracy"]},
    ])
    sns.barplot(data=df_comp, x="Metric", y="Score", hue="Method", palette=["#7f8c8d", "#2ecc71"], ax=ax1)
    ax1.set_title("Panel A: NLP Greenwashing Detection Performance", fontweight="bold")
    ax1.set_ylim(0.5, 1.05)
    ax1.set_ylabel("Evaluation Score")
    for p in ax1.patches:
        h = p.get_height()
        if h > 0:
            ax1.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                         ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontsize=9)

    # Panel B: PCA Semantic Projection of CSR Disclosures
    ax2 = axes[1]
    pca = PCA(n_components=2, random_state=42)
    sample_sub = np.random.choice(len(embeddings), size=min(1200, len(embeddings)), replace=False)
    emb_2d = pca.fit_transform(embeddings[sample_sub])
    sub_labels = labels[sample_sub]

    scatter = ax2.scatter(
        emb_2d[:, 0], emb_2d[:, 1],
        c=sub_labels, cmap="coolwarm", alpha=0.6, s=25, edgecolors="none"
    )
    ax2.set_title(f"Panel B: Transformer Embedding Space (PCA Var Explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontweight="bold")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    cbar = plt.colorbar(scatter, ax=ax2, ticks=[0, 1])
    cbar.set_ticklabels(["Factual / Audited", "High Greenwash Rhetoric"])

    plt.suptitle("Figure 8: Transformer Multi-Head Self-Attention vs. Lexical TF-IDF in ESG Greenwashing", y=1.03, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    import json
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "esg_financial_panel.parquet")
    df = pd.read_parquet(data_path)

    # Greenwashing binary label: 1 if high greenwashing gap (> 0.20), 0 otherwise
    labels = (df["greenwashing_gap"] > 0.20).astype(int).values
    texts = df["csr_disclosure"].tolist()

    train_idx = np.where(df["year"] <= 2022)[0]
    test_idx = np.where(df["year"] > 2022)[0]

    m_tfidf, m_tf, embeddings = compare_transformer_vs_tfidf(texts, labels, train_idx, test_idx)

    logger.info("\n--- NLP BENCHMARK RESULTS ---")
    logger.info(f"TF-IDF:      {m_tfidf}")
    logger.info(f"Transformer: {m_tf}")

    out_metrics = {
        "TF_IDF_Baseline": m_tfidf,
        "Transformer_Self_Attention": m_tf
    }
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics", "transformer_vs_tfidf_nlp.json")
    with open(metrics_path, "w") as f:
        json.dump(out_metrics, f, indent=4)

    fig_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures", "figure8_transformer_nlp_greenwash.png")
    plot_transformer_comparison(m_tfidf, m_tf, embeddings, labels, fig_path)
    logger.info(f"Saved Figure 8 to {fig_path}")
