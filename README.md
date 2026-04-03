# 🇦🇿 Azerbaijani Sentiment Analysis Pipeline

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Gensim](https://img.shields.io/badge/Gensim-Word2Vec%2FFastText-4CAF50?style=for-the-badge)](https://radimrehurek.com/gensim/)
[![YouTube API](https://img.shields.io/badge/YouTube%20Data%20API-v3-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://developers.google.com/youtube/v3)

**A production-grade, end-to-end NLP pipeline for 3-class sentiment analysis in Azerbaijani.**  
*From raw YouTube comments to domain-aware GRU models — with rigorous linguistic filtering, embedding comparison, and cross-domain generalization experiments.*

</div>

---

## 📊 Results at a Glance

| Model | Embedding | Macro-F1 | Accuracy |
|-------|-----------|----------|----------|
| GRU | Word2Vec (Frozen) | 0.6503 | 65.3% |
| GRU | Word2Vec (Fine-tuned) | 0.7712 | 77.4% |
| GRU | FastText (Frozen) | 0.6536 | 65.6% |
| **GRU** | **FastText (Fine-tuned)** | **0.7827** ✨ | **78.4%** |

> Best model: **FastText Fine-tuned GRU** — `Macro-F1: 0.7827` on a 19,960-sample held-out test set across 5 domains.

---

## 🏗️ Architecture Overview

```
Raw Datasets (Part 1)          YouTube Comments (Part 2)
        │                               │
        ▼                               ▼
  ┌──────────────┐             ┌──────────────────┐
  │ ETL Pipeline │             │  Azerbaijani     │
  │ (5 datasets) │             │  Language Filter │
  │ deduplicate  │             │  (2-layer score) │
  └──────┬───────┘             └────────┬─────────┘
         │                              │
         ▼                              ▼
  ┌─────────────────────────────────────────┐
  │         Full Corpus (Unlabeled)         │
  │   Word2Vec (300d) │ FastText (300d)     │
  │   sg=1 | window=5 | min_count=3        │
  └────────────────┬────────────────────────┘
                   │  Embedding Matrix
                   ▼
  ┌────────────────────────────────────────────┐
  │            GRU Classifier                  │
  │  Input → Embedding → GRU(64) →            │
  │  LayerNorm → Dropout(0.3) → Dense(3)      │
  │  Loss: Focal Loss (γ=2.0, α=0.25)         │
  │  Optimizer: AdamW + ReduceLROnPlateau      │
  └────────────────────────────────────────────┘
```

---

## 🔬 Key Technical Contributions

### 1. Two-Layer Azerbaijani Language Filter
Distinguishing Azerbaijani from Turkish is uniquely hard due to ~60% lexical overlap. This pipeline implements a **scored heuristic filter** with two independent signals:

- **Layer A — Orthographic:** Character `ə` (schwa) carries `+4.0` weight — the single strongest AZ signal, absent from Turkish.
- **Layer B — Lexical:** Positive markers (`mən`, `deyil`, `üçün`) vs Turkish penalties (`ben`, `değil`, `-yor` suffix).
- **Acceptance threshold:** Score ≥ 2.0 (validated manually on 50 samples).

### 2. Multi-Source Data Fusion
Combines 5 heterogeneous labeled datasets from Part 1 with 50,000+ unlabeled YouTube comments from Part 2 for joint embedding training. A **strict deduplication layer** (content hashing) removed 41,102 cross-dataset duplicate rows, preventing catastrophic data leakage that would have inflated test accuracy to ~99%.

### 3. Focal Loss for Class Imbalance
The Neutral class is chronically underrepresented in Azerbaijani sentiment data. Rather than naïve class weighting, the pipeline uses **Focal Loss** (γ=2.0) which down-weights easy examples and focuses the model on hard-to-classify Neutral instances. Complemented by targeted text augmentation (word dropout at p=0.15).

### 4. Domain Shift (Leave-One-Out) Analysis
For each of the 5 domains, a separate model is trained on the remaining 4 and evaluated on the held-out domain — quantifying how much domain-specific knowledge the model relies on. This is a standard evaluation protocol in production NLP systems.

---

## 📁 Project Structure

```
nlp-ödev/
│
├── 📄 Part 1 — Data Processing & Embeddings
│   ├── process_assignment.py        # ETL: 5 raw datasets → cleaned Excel + corpus.txt
│   ├── optuna_tune_embeddings.py    # Hyperparameter search for W2V / FastText
│   ├── train_embeddings.py          # Final embedding training with Optuna best params
│   ├── evaluate_embeddings.py       # Embedding quality evaluation (analogy, similarity)
│   └── eval_utils.py                # Shared corpus loading utilities
│
├── 🤖 Part 2 — Modeling & Evaluation
│   └── part2_code/
│       ├── part2_modeling.py        # Main pipeline: train, evaluate, domain shift
│       ├── part2_utils.py           # Azerbaijani language filter (is_azerbaijani)
│       ├── assign_domains.py        # Keyword-based domain classifier (word-boundary regex)
│       ├── collect_youtube_data.py  # YouTube Data API v3 comment collector
│       └── fetch_metadata_retroactive.py  # Video metadata enrichment
│
├── 📊 Outputs (generated, not committed)
│   ├── embeddings/                  # Word2Vec + FastText models (~2.8 GB)
│   ├── cleaned_data/                # Processed Excel files from Part 1
│   ├── part1_with_domains/          # Domain-labeled Part 1 data
│   ├── part2_data/                  # Collected YouTube comments
│   ├── experiment_plots/            # 16 evaluation plots (confusion matrices, F1 bars)
│   └── best_sentiment_model.keras   # Best trained GRU model (~1.3 GB)
│
├── 📈 Reports
│   ├── Final_Report.md              # Full technical report with analysis
│   ├── final_experiment_report.txt  # Raw metrics (auto-generated by pipeline)
│   └── model_diagram.md             # Mermaid architecture diagram
│
└── 🛠️ Utilities
    └── regenerate_plots.py          # Standalone script to regenerate all 16 plots
```

---

## 🚀 Quickstart

### Prerequisites

```bash
pip install tensorflow gensim scikit-learn pandas openpyxl seaborn matplotlib
pip install google-api-python-client ftfy optuna
```

> **GPU recommended.** The pipeline uses `tf.keras.mixed_precision` (float16).
> Tested on Python 3.10, TensorFlow 2.14.

### Environment Setup

```bash
# Required for YouTube comment collection only
export YOUTUBE_API_KEY="your_key_here"
```

### Run the Full Pipeline

```bash
# ── Part 1: Data Processing ──────────────────────────────────────────────────
# Step 1: Clean raw datasets → cleaned_data/ + corpus_all.txt
python process_assignment.py

# Step 2: (Optional) Tune embedding hyperparameters with Optuna
python optuna_tune_embeddings.py

# Step 3: Train final Word2Vec + FastText models
python train_embeddings.py

# Step 4: Evaluate embedding quality
python evaluate_embeddings.py

# ── Part 2: Data Collection and Modeling ─────────────────────────────────────
# Step 5: Assign domains to Part 1 data
python part2_code/assign_domains.py

# Step 6: (Skip if you have part2_data/) Collect YouTube comments
python part2_code/collect_youtube_data.py

# Step 7: Train all 4 GRU variants + run domain/shift evaluation
python part2_code/part2_modeling.py

# ── Visualization ─────────────────────────────────────────────────────────────
# Regenerate all 16 plots from existing report file
python regenerate_plots.py
```

---

## 📈 Experiment Plots

All plots are generated automatically and saved to `experiment_plots/`. Key visualizations:

| Plot | Description |
|------|-------------|
| `f1_comparison.png` | Macro-F1 bar chart across all 4 embedding variants |
| `cm_FT_Tuned.png` | Confusion matrix for the best model |
| `cm_Domain_*.png` | Per-domain confusion matrices (5 domains) |
| `cm_Shift_Test_*.png` | Domain shift confusion matrices (5 held-out experiments) |
| `domain_sentiment_distribution.png` | YouTube comment sentiment breakdown per domain |

---

## 🌐 Domain Coverage

The pipeline collects and evaluates sentiment across 5 Azerbaijani domains:

| Domain | YouTube Keywords (sample) | Part 1 Source |
|--------|--------------------------|---------------|
| Technology & Digital Services | `xiaomi azərbaycan`, `bakcell tarifləri` | `labeled-sentiment.xlsx` |
| Finance & Business | `kredit faizləri`, `manat dollar məzənnəsi` | `merged_dataset_CSV__1_.xlsx` |
| Social Life & Entertainment | `meyxana`, `yeni mahnilar 2024` | `merged_dataset_CSV__1_.xlsx` |
| Retail & Lifestyle | `28 mall`, `endirimler` | `train-00000-of-00001.xlsx` |
| Public Services | `asan xidmət`, `pensiya artımı` | `test__1_.xlsx` + `train__3_.xlsx` |

---

## ⚙️ Model Configuration

```python
# Architecture
GRU_UNITS       = 64
DROPOUT_RATE    = 0.3
NUM_CLASSES     = 3        # Negative / Neutral / Positive
MAX_TOKENS      = 100      # Max sequence length (tokens)
EMBEDDING_DIM   = 300      # Word2Vec & FastText vector size

# Training
BATCH_SIZE      = 128
EPOCHS          = 15       # With EarlyStopping (patience=3)
OPTIMIZER       = AdamW(lr=1e-3, weight_decay=0.01)
LOSS            = FocalLoss(gamma=2.0, alpha=0.25)
LR_SCHEDULE     = ReduceLROnPlateau(factor=0.5, patience=2)

# Embeddings
W2V_PARAMS      = dict(sg=1, window=5, min_count=3, negative=10, epochs=10)
FT_PARAMS       = dict(sg=1, window=5, min_count=3, min_n=3, max_n=6)
```

---

## 🔒 Privacy & Compliance

- **YouTube API:** All data collection respects the [YouTube ToS §IV.J](https://www.youtube.com/t/terms). Raw comment data is not redistributed in this repository.
- **API Keys:** Retrieved via `os.getenv("YOUTUBE_API_KEY")` — never hardcoded.
- **Personal Data:** `video_metadata_all.xlsx` and `part2_data/` are excluded from version control via `.gitignore`.

---

## 📚 References

- [Word2Vec: Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Mikolov et al., 2013
- [FastText: Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — Bojanowski et al., 2017
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) — Lin et al., 2017 (RetinaNet)
- [Gated Recurrent Unit (GRU)](https://arxiv.org/abs/1412.3555) — Cho et al., 2014

---

## 👤 Author

**Mehmet Ali Yılmaz** — Student ID: 21050111057  
CENG442 — Natural Language Processing  
4th Year, 1st Semester
