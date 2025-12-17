## Azerbaijani Text Preprocessing & Embeddings

This project implements a robust, domain-aware text preprocessing pipeline for Azerbaijani. It processes, cleans, and standardizes five sentiment-annotated datasets. Finally, it trains and evaluates Word2Vec and FastText embedding models on the cleaned corpus.

**Members:**
* `Mehmet Ali Yılmaz`
* `Cemilhan Sağlam`
* `Muhammed Esat Çelebi`


---

## 1. Data & Project Goal

The primary goal of this project was to process five disparate Azerbaijani text datasets (Excel files) and create a standardized, clean corpus suitable for sentiment analysis.

The datasets varied in their annotation schemes, using binary (0/1) or 3-class (Negative/Neutral/Positive) labels. A key task was to unify these into a single numerical format. The 3-class polarity was preserved (`Negative=0.0`, `Neutral=0.5`, `Positive=1.0`). Retaining the `0.5` neutral class, rather than discarding it, provides a richer dataset, enabling future models to distinguish between mild dissatisfaction and strong negativity, or even to treat sentiment as a regression problem.

The pipeline processed **124,051** total rows after cleaning and deduplication.

## 2. Preprocessing Pipeline

A multi-stage preprocessing pipeline was developed to normalize and clean the raw text. The key transformation rules included:

* **Azerbaijani-Aware Casing:** Standard `.lower()` is insufficient for Turkic languages. A custom function was used to correctly map `İ`→`i` and `I`→`ı` *before* lowercasing.
* **Entity Normalization:** Common web entities were not deleted but standardized into special tokens (e.g., URLs → `<URL>`, emails → `<EMAIL>`, user mentions → `<USER>`).
* **Noise & HTML Removal:** HTML tags and excess punctuation/whitespace were stripped.
* **Digit Normalization:** All numerical digits were mapped to a single `<NUM>` token.
* **Character Repetition:** Repeated characters (e.g., `cooool`) were collapsed to a maximum of two (`cool`).

### Before-After Example

A single processing function (`normalize_text_az`) applies these rules sequentially.

* **Before:** `Mükəmməl bir film... #Super 🤩 https://example.com qiyməti 20 AZN idi!!`
* **After (Base):** `mükəmməl bir film super <EMO_POS> <URL> qiyməti <NUM> azn idi`
* **After (Domain-Aware):** `mükəmməl bir film super <EMO_POS> <URL> qiyməti <PRICE> idi`

## 3. Mini Challenges Implemented

Several "mini-challenges" from the assignment brief were implemented to enhance the pipeline's robustness.

* **Negation Handling:** The pipeline marks the scope of negation. After encountering a negator (e.g., `yox`, `deyil`), the next three tokens are appended with a `_NEG` suffix (e.g., `yaxşı` becomes `yaxşı_NEG`). This helps the model learn that `yaxşı` and `yaxşı_NEG` are semantically opposite.
* **Emoji Mapping:** A small dictionary maps common positive/negative emojis to `<EMO_POS>` and `<EMO_NEG>` tokens (e.g., `😊` → `<EMO_POS>`, `😞` → `<EMO_NEG>`).
* **De-asciification & Slang:** A small dictionary normalizes common "de-asciified" words and slang (e.g., `cox` → `çox`, `yaxsi` → `yaxşı`).
* **Hashtag Splitting:** CamelCase hashtags are split into their constituent words (e.g., `#QarabagIsBack` → `qarabag is back`).

## 4. Domain-Aware Normalization

The pipeline is "domain-aware," meaning its cleaning rules adapt to the text's content.

1.  **Detection:** A regex-based function (`detect_domain`) classifies each raw text into one of four domains: `news`, `social`, `reviews`, or `general`.
2.  **Normalization:** Text classified as `reviews` undergoes an additional normalization step. Specific patterns like `20 azn` or `5 ulduz` are converted into generalized tokens (`<PRICE>`, `<STARS_5>`). This allows the model to learn the *concept* of price or a high rating, rather than memorizing specific numbers.
3.  **Tagging:** For the final `corpus_all.txt` file, each sentence is prepended with its detected domain tag (e.g., `domreviews ...`, `domnews ...`).

## 5. Embeddings: Training & Evaluation

Two embedding models were trained on the combined 124,051 cleaned documents from the `cleaned_data/*.xlsx` files.

### Training Settings

| Parameter | Word2Vec (Skip-gram) | FastText (Skip-gram) |
| :--- | :--- | :--- |
| `vector_size` | 300 | 300 |
| `window` | 5 | 5 |
| `min_count` | 3 | 3 |
| `epochs` | 10 | 10 |
| `sg` | 1 (Skip-gram) | 1 (Skip-gram) |
| `min_n` | N/A | 3 |
| `max_n` | N/A | 6 |

### Quantitative Evaluation

**Lexical Coverage:**
Coverage (the ratio of in-vocabulary words) was identical for both models (~93-99% per file, e.g., `0.932` vs `0.932`). This is expected, as both used the same `min_count=3` threshold. This metric is unsuitable for comparing the two, as it fails to measure FastText's primary advantage: generating vectors for Out-of-Vocabulary (OOV) words.

**Semantic Similarity:**
Models were tested on their ability to differentiate synonyms (high score is better) and antonyms (low score is better).

| Metric | Word2Vec | FastText | Ideal |
| :--- | :---: | :---: | :---: |
| Synonym Similarity | 0.360 | **0.439** | High |
| Antonym Similarity | **0.265** | 0.333 | Low |
| **Separation (Syn - Ant)** | 0.095 | **0.106** | **High** |

FastText demonstrated a slightly better, though not decisive, ability to group synonyms and separate antonyms.

### Qualitative Evaluation (Nearest Neighbors)

The qualitative analysis of nearest neighbors revealed the most significant differences between the models.

| Seed Word | Word2Vec Neighbors | FastText Neighbors | Analysis |
| :--- | :--- | :--- | :--- |
| **`yaxşı`** (good) | `['<RATING_POS>', 'iyi', 'yaxshi']` | `['yaxşıı', 'yaxşıkı', 'yaxşıca']` | **W2V** found *semantic* (contextual) neighbors, including the domain token `<RATING_POS>`. **FT** found *morphological* (structural) neighbors and typos. |
| **`pis`** (bad) | `['vərdişlərə', 'günd', 'yaxşıdır_NEG']` | `['piis', 'pisdii', 'pi', 'pisə', 'pisleşdi']` | **Critical Failure for W2V.** Word2Vec failed to learn a meaningful representation for this key sentiment word. **Critical Win for FT.** FastText's sub-word model correctly identified typos and related morphological variations. |
| **`bahalı`** (expensive) | `['metallarla', 'portretlerinə', 'radiusda']` | `['bahalıı', 'bahalısı', 'bahalıq']` | **W2V** found weak contextual links. **FT** found strong *structural* links, making it robust to typos like 'pahalı' or 'bahalıı'. |
| **`ucuz`** (cheap) | `['düzəltdirilib', 'şeytanbazardan', 'sorbasi']` | `['ucuzu', 'ucuza', 'ucuzdu']` | Similar to `bahalı`, FastText proves superior at handling morphological variations (`ucuzu`, `ucuza`) common in real-world data. |
| **`yox`** (no/negator) | `['idi_NEG', 'olur_NEG', 'imiş_NEG']` | `['yoxhjgsjsh', 'yoxh', 'idi_NEG']` | **Success for W2V.** This result proves the negation-handling mini-challenge was successful. W2V perfectly grouped the `yox` token with other tokens that were tagged as `_NEG`. |
| **`<RATING_POS>`** | `['süper', 'deneyin', 'yaradı']` | `['<RATING_NEG>', 'süperr', 'süper']` | **Success for Both.** Both models learned that the token is related to positive words. **FT** performed slightly better by *also* identifying its direct antonym, `<RATING_NEG>`, as a close neighbor. |

## 7. Reproducibility

The project is divided into three distinct, runnable scripts.

1.  **`process_assignment.py`**: Reads the raw Excel files from the root directory, performs all cleaning and domain-aware normalization, and saves the 5 output files to `cleaned_data/` and the main `corpus_all.txt`.
2.  **`train_embeddings.py`**: Reads the 5 Excel files from `cleaned_data/`, trains the Word2Vec and FastText models, and saves the final models to the `embeddings/` directory.
3.  **`evaluate_embeddings.py`**: Loads the trained models from `embeddings/` and runs the quantitative and qualitative analyses, printing the results to the console.

**To run this project:**
1.  Ensure all dependencies are installed: `pip install pandas openpyxl regex ftfy gensim numpy`
2.  Place the 5 raw dataset `.xlsx` files in the repository's root folder.
3.  Run the scripts in sequence:
    ```bash
    python process_assignment.py
    python train_embeddings.py
    python evaluate_embeddings.py
    ```

## 8. Conclusions

This analysis concludes that **FastText is the superior model for this specific task and dataset.**

**Reasoning:**
1.  **Robustness to Errors:** FastText successfully learned meaningful vectors for critical sentiment words (e.g., `pis`) where Word2Vec failed. Its sub-word architecture makes it inherently robust to the typos (`piis`, `yaxşıı`) common in social media and review data.
2.  **Morphological Richness:** Azerbaijani is an agglutinative (suffix-heavy) language. FastText's ability to model sub-words allows it to understand the relationship between `ucuz`, `ucuzu`, and `ucuza`, whereas Word2Vec treats them as entirely separate, unrelated tokens.
3.  **Semantic Performance:** While Word2Vec showed excellent *contextual* understanding (e.g., linking `yox` to `_NEG` tokens), FastText's robustness to data noise and morphology, combined with its slightly better quantitative "Separation Score," makes it the more reliable and effective model.
