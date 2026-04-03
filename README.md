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

```text
Raw Datasets (Phase 1)          YouTube Comments (Phase 2)
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
  │   sg=1 | window=5 | min_count=3         │
  └────────────────┬────────────────────────┘
                   │  Embedding Matrix
                   ▼
  ┌────────────────────────────────────────────┐
  │            GRU Classifier                  │
  │  Input → Embedding → GRU(64) →             │
  │  LayerNorm → Dropout(0.3) → Dense(3)       │
  │  Loss: Focal Loss (γ=2.0, α=0.25)          │
  │  Optimizer: AdamW + ReduceLROnPlateau      │
  └────────────────────────────────────────────┘
```

---

## 🧹 Phase 1: Robust Preprocessing Pipeline

The primary goal of Phase 1 was to process five disparate Azerbaijani text datasets (Excel files) and create a standardized, clean corpus. The pipeline processed **124,051** total rows after cleaning and deduplication.

### Key Transformation Rules
* **Azerbaijani-Aware Casing:** Standard `.lower()` is insufficient for Turkic languages. A custom function correctly maps `İ`→`i` and `I`→`ı` *before* lowercasing.
* **Entity & Digit Normalization:** Standardized into special tokens (`<URL>`, `<EMAIL>`, `<USER>`, `<NUM>`).
* **Negation Handling:** Evaluates scope of negation. After encountering a negator (e.g., `yox`, `deyil`), the next three tokens receive a `_NEG` suffix (e.g., `yaxşı_NEG`) to maintain semantic opposition.
* **Emoji & Slang Mapping:** Positive/negative emojis are converted to `<EMO_POS>` / `<EMO_NEG>`. Common "de-asciified" slang is corrected (`cox` → `çox`).

### Before & After Examples
* **Before:** `Mükəmməl bir film... #Super 🤩 https://example.com qiyməti 20 AZN idi!!`
* **After (Base):** `mükəmməl bir film super <EMO_POS> <URL> qiyməti <NUM> azn idi`
* **After (Reviews Domain):** `mükəmməl bir film super <EMO_POS> <URL> qiyməti <PRICE> idi`

---

## 🔬 Phase 2: Domain-Aware Modeling & Embeddings

### 1. Two-Layer Azerbaijani Language Filter
Distinguishing Azerbaijani from Turkish is uniquely hard due to ~60% lexical overlap. This pipeline implements a **scored heuristic filter**:
- **Layer A — Orthographic:** Character `ə` (schwa) carries `+4.0` weight — the single strongest AZ signal, absent from Turkish.
- **Layer B — Lexical:** Positive markers (`mən`, `deyil`, `üçün`) vs Turkish penalties (`ben`, `değil`, `-yor` suffix).
- **Acceptance threshold:** Score ≥ 2.0.

### 2. FastText vs Word2Vec Embedding Analysis
Evaluations proved **FastText** as the superior model for Azerbaijani due to its sub-word architecture resolving rich morphology and typos.

**Semantic Similarity Separation (Synonym - Antonym)**
| Metric | Word2Vec | FastText | Ideal |
| :--- | :---: | :---: | :---: |
| Synonym Similarity | 0.360 | **0.439** | High |
| Antonym Similarity | **0.265** | 0.333 | Low |
| **Separation (Syn - Ant)** | 0.095 | **0.106** | **High** |

**Qualitative Evaluation (Nearest Neighbors)**
| Seed Word | Word2Vec Neighbors | FastText Neighbors | Analysis |
| :--- | :--- | :--- | :--- |
| **`pis`** (bad) | `['vərdişlərə', 'günd', 'yaxşıdır_NEG']` | `['piis', 'pisdii', 'pi', 'pisə', 'pisleşdi']` | **Critical Win for FastText.** Identified typos and related morphological variations. Word2Vec completely failed on this key sentiment word. |
| **`ucuz`** (cheap) | `['düzəltdirilib', 'şeytanbazardan']` | `['ucuzu', 'ucuza', 'ucuzdu']` | FastText effectively handles agglutinative morphological variations (`ucuzu`, `ucuza`). |
| **`yox`** (no) | `['idi_NEG', 'olur_NEG', 'imiş_NEG']` | `['yoxhjgsjsh', 'yoxh', 'idi_NEG']` | **Success for Word2Vec.** Perfectly grouped `yox` with other `_NEG` tagged tokens. |

### 3. Focal Loss & Domain Shift (Leave-One-Out)
Due to a high imbalance of the "Neutral" sentiment class:
- **Focal Loss** (γ=2.0) down-weights easy examples and focuses the model on hard-to-classify Neutral instances. Complemented by word dropout (p=0.15).
- **Domain Shift Validation:** For each of the 5 domains, a separate model is trained on the remaining 4 and evaluated on the held-out domain, stress-testing its generalization.

---

## 📁 Project Structure

```
azerbaycani-nlp-pipeline/
│
├── 📄 Phase 1 — Data Processing & Embeddings
│   ├── process_datasets.py        # ETL: raw datasets → cleaned Excel + corpus.txt
│   ├── optuna_tune_embeddings.py    # Hyperparameter search for W2V / FastText
│   ├── train_embeddings.py          # Final embedding training with Optuna params
│   ├── evaluate_embeddings.py       # Embedding quality evaluation (analogy, similarity)
│   └── eval_utils.py                # Shared corpus loading utilities
│
├── 🤖 Phase 2 — Modeling & Evaluation
│   └── part2_code/
│       ├── part2_modeling.py        # Main pipeline: train, evaluate, domain shift
│       ├── part2_utils.py           # Azerbaijani language filter (is_azerbaijani)
│       ├── assign_domains.py        # Keyword-based domain classifier
│       ├── collect_youtube_data.py  # YouTube Data API v3 comment collector
│       └── fetch_metadata_retroactive.py  # Video metadata enrichment
│
├── 📊 Outputs (generated locally)
│   ├── embeddings/                  # Word2Vec + FastText models (~2.8 GB)
│   ├── cleaned_data/                # Processed Excel files
│   ├── part1_with_domains/          # Domain-labeled data
│   ├── part2_data/                  # Collected YouTube comments
│   ├── experiment_plots/            # 16 standard evaluation plots
│   └── best_sentiment_model.keras   # Best trained GRU model
│
├── 📈 Reports
│   ├── Final_Report.md              # Full technical analytical report
│   └── final_experiment_report.txt  # Raw metrics auto-generated by pipeline
│
└── 🛠️ Utilities
    └── regenerate_plots.py          # Standalone script to redraw all plots
```

---

## 🚀 Quickstart

**Prerequisites:**
```bash
pip install tensorflow gensim scikit-learn pandas openpyxl seaborn matplotlib google-api-python-client ftfy optuna
```

**Run the Full Pipeline:**
```bash
# Data Processing & Embeddings
python process_datasets.py
python optuna_tune_embeddings.py
python train_embeddings.py
python evaluate_embeddings.py

# Data Collection & GRU Modeling
export YOUTUBE_API_KEY="your_key_here"  # Required for data collection step
python part2_code/assign_domains.py
python part2_code/collect_youtube_data.py
python part2_code/part2_modeling.py

# Re-draw existing plots
python regenerate_plots.py
```

---

## 🔒 Privacy & Compliance
- **YouTube API:** Collection conforms to API strictly. Raw data (`part2_data/`) is not redistributed.
- **Git:** All personal and proprietary datasets, large binaries (`.keras`), and `.env` credentials are systematically excluded via `.gitignore`.

---

## 📚 References
- [Word2Vec: Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013)
- [FastText: Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) (Bojanowski et al., 2017)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) (Lin et al., 2017)
- [Gated Recurrent Unit (GRU)](https://arxiv.org/abs/1412.3555) (Cho et al., 2014)

---

## 👤 Author

Mehmet Ali Yılmaz — Natural Language Processing Pipeline  
