"""
CENG442 - NLP Assignment Part 2
Module: part2_modeling.py
Description:
    This is the core modeling pipeline for the Azerbaijani Sentiment Analysis project.
    It integrates heterogenous data sources (Part 1 Labeled + Part 2 Unlabeled) to 
    train and evaluate Gated Recurrent Unit (GRU) models.

    Key Features:
    - Embedding Strategy: Comparative analysis of Word2Vec vs FastText (Frozen vs Fine-Tuned).
    - Architecture: Single-layer GRU with regularized Dropout, optimized for short-text sentiment.
    - Experimentation: 
        1. Base Performance (Macro-F1)
        2. Domain-Specific Analysis (5 Verbatim Domains)
        3. Domain-Shift Generalization (Leave-One-Group-Out Cross-Validation)
    
    The pipeline is designed for reproducibility, utilizing seed-based initialization
    and strict data deduplication to prevent target leakage.
"""
import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from gensim.models import Word2Vec, FastText
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import backend as K

# =================================================================
# REPRODUCIBILITY & GPU OPTIMIZATION
# =================================================================
# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Enable mixed precision for faster training on compatible GPUs
# This uses float16 for computation while keeping float32 for variables
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled (float16)")
except Exception as e:
    print(f"Mixed precision not available: {e}")

# =================================================================
# FOCAL LOSS IMPLEMENTATION (For Class Imbalance)
# =================================================================
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for multi-class classification.
    Focuses training on hard-to-classify examples (like the underrepresented Neutral class).
    
    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples. Default: 2.0
        alpha: Class balancing weight. Default: 0.25
        
    Returns:
        A loss function compatible with Keras model.compile().
        
    Reference:
        Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Convert sparse labels to one-hot for multi-class focal loss
        # FIX: Use tf.reshape instead of conditional tf.squeeze for graph mode compatibility
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true_int, depth=NUM_CLASSES)
        
        # Calculate focal weights: (1 - p_t)^gamma
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        # Cross entropy component
        ce = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)
        
        return tf.reduce_mean(alpha * focal_weight * ce)
    return focal_loss_fixed

# =================================================================
# CONFIGURATION
# =================================================================
# ... (Previous Mappings Remain) ...
# DOMAIN_MAPPING removed - domains are now embedded in the files
DATA_DIR_PART1 = "part1_with_domains"

MAX_TOKENS = 100
EMBEDDING_DIM = 300 
GRU_UNITS = 64
DROPOUT_RATE = 0.3
NUM_CLASSES = 3

REPORT_FILE = "final_experiment_report.txt"
PLOT_DIR = "experiment_plots"

if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# =================================================================
# TEXT CLEANING
# =================================================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text) 
    return text

# =================================================================
# DATA AUGMENTATION (For Minority Class Enhancement)
# =================================================================
def augment_text(text, drop_prob=0.15, swap_prob=0.1):
    """
    Simple text augmentation via random word dropout and swap.
    
    Args:
        text: Input text string.
        drop_prob: Probability of dropping each word. Default: 0.15
        swap_prob: Probability of swapping adjacent words. Default: 0.1
        
    Returns:
        Augmented text string.
    """
    words = text.split()
    if len(words) < 3:
        return text
    
    # Random word dropout
    words = [w for w in words if random.random() > drop_prob]
    
    # Random adjacent word swap
    for i in range(len(words) - 1):
        if random.random() < swap_prob:
            words[i], words[i+1] = words[i+1], words[i]
    
    return ' '.join(words)

# =================================================================
# TRAINING CALLBACKS (LR Scheduling + Early Stopping)
# =================================================================
def get_training_callbacks():
    """
    Returns a list of Keras callbacks for optimized training:
    - EarlyStopping: Stops training when validation loss stops improving.
    - ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        min_delta=0.002,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
    
    return [early_stop, lr_scheduler]

# =================================================================
# REPORTING & PLOTTING UTILS
# =================================================================
def save_text_report(content):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(content + "\n")

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Neg', 'Neu', 'Pos'], 
                yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    safe_title = re.sub(r"\W+", "_", title)
    plt.savefig(os.path.join(PLOT_DIR, f"cm_{safe_title}.png"))
    plt.close()

def plot_f1_comparison(results_dict):
    names = list(results_dict.keys())
    scores = list(results_dict.values())
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=names, y=scores, hue=names, palette="viridis", legend=False)
    plt.title("Macro-F1 Score Comparison")
    plt.ylim(0, 1.0)
    plt.ylabel("Macro-F1")
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "f1_comparison.png"))
    plt.close()

def plot_domain_predictions(df_yt):
    # Stacked Bar Chart for Domain Sentiment Distribution
    cross_tab = pd.crosstab(df_yt['domain'], df_yt['pred'], normalize='index') * 100
    ax = cross_tab.plot(kind='bar', stacked=True, color=['#e74c3c', '#95a5a6', '#2ecc71'], figsize=(10, 6))
    
    plt.title("Sentiment Distribution per Domain (Predicted)")
    plt.ylabel("Percentage")
    plt.legend(["Negative", "Neutral", "Positive"], loc='upper left', bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "domain_sentiment_distribution.png"))
    plt.close()

# =================================================================
# MODEL BUILDER
# =================================================================
def build_embedding_matrix(word_index, keyed_vectors, num_words):
    """
    Constructs a weight matrix for the Keras Embedding layer using pre-trained vectors.
    
    Args:
        word_index (dict): Keras tokenizer's word-to-index map.
        keyed_vectors (KeyedVectors): Gensim model's word vectors (W2V or FT).
        num_words (int): Velocity size cap.
        
    Returns:
        tuple: (embedding_matrix, oov_rate)
        
    Implementation Note:
        - The matrix is initialized with a random normal distribution (scale=0.02) rather than zeros.
          This ensures that OOV tokens have a non-zero, trainable starting point 
          (especially important for the Fine-Tuning experiments).
        - Padding (index 0) is explicitly forced to zero.
    """
    embedding_dim = keyed_vectors.vector_size
    matrix = np.random.normal(scale=0.02, size=(num_words, embedding_dim)) # Better Initialization
    matrix[0] = np.zeros(embedding_dim) # Padding must be zero-vector
    
    hits = 0
    misses = 0
    for word, idx in word_index.items():
        if idx >= num_words:
            continue
        try:
            matrix[idx] = keyed_vectors[word]
            hits += 1
        except KeyError:
            misses += 1
            # Already random initialized, so we just count the miss
            
    oov_rate = misses / max(hits + misses, 1)
    return matrix, oov_rate

def build_gru_model(vocab_size, embedding_dim, max_len, embedding_matrix, trainable):
    """
    Compiles an RNN architecture for sentiment classification.
    
    Architecture:
        Input -> Embedding -> GRU (64 units) -> LayerNormalization -> Dropout -> Dense (Softmax)
    """
    inp = Input(shape=(max_len,))
    emb = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=trainable,
        mask_zero=True
    )(inp)
    
    # Standard GRU layer
    x = GRU(GRU_UNITS)(emb)
    
    # Dropout for regularization
    x = LayerNormalization()(x)
    
    x = Dropout(DROPOUT_RATE)(x)
    out = Dense(NUM_CLASSES, activation="softmax")(x)
    
    model = Model(inp, out)
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=0.01),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy"]
    )
    return model

# =================================================================
# EVALUATION UTILS
# =================================================================
def evaluate_model(model, X_test, y_test, title=""):
    print(f"\n--- Evaluation: {title} ---")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, digits=4)
    
    # Console Output
    print(f"Macro-F1: {macro_f1:.4f}")
    print(report)
    
    # Save to File
    header = f"\n{'='*40}\nEVALUATION: {title}\n{'='*40}\n"
    save_text_report(header + f"Macro-F1: {macro_f1:.4f}\n\n" + report)
    
    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, title)
    
    return macro_f1

# =================================================================
# MAIN PIPELINE
# =================================================================
# =================================================================
# SENTIMENT MAPPING (PART 1 TO PART 2)
# =================================================================
def map_sentiment_part1(v):
    """
    Standardizes Part 1 labels to Part 2 scheme:
    Target: 0=Negative, 1=Neutral, 2=Positive
    
    IMPORTANT MAPPING LOGIC:
    ========================
    Part 1 datasets use inconsistent label schemes:
    - Some use strings: "pos", "neg", "neu"
    - Some use integers: 0, 1, 2 (where meaning varies!)
    
    This function unifies them:
    - String "pos"/"positive"/"1" or Azerbaijani "müsbət" -> 2 (Positive)
    - String "neu"/"neutral"/"2" or "neytral" -> 1 (Neutral)
    - String "neg"/"negative"/"0" or "mənfi" -> 0 (Negative)
    - Integer 0 -> 0 (Negative)
    - Integer 1 -> 2 (Positive) [Binary sentiment: 1=Positive]
    - Integer 2 -> 1 (Neutral) [Tri-class: 2=Neutral in some datasets]
    """
    s = str(v).strip().lower()
    if s in ("pos", "positive", "1", "müsbət", "good", "pozitiv"): return 2
    if s in ("neu", "neutral", "2", "neytral"): return 1
    if s in ("neg", "negative", "0", "mənfi", "bad", "negativ"): return 0
    
    # Try direct int conversion as fallback
    try:
        val = int(v)
        if val == 0: return 0
        if val == 1: return 2  # Binary sentiment: 1 = Positive
        if val == 2: return 1  # Tri-class: 2 = Neutral
    except:
        pass
        
    return 1  # Default to Neutral if unknown

# =================================================================
# MAIN PIPELINE
# =================================================================
def main():
    # Clear previous report
    if os.path.exists(REPORT_FILE):
        os.remove(REPORT_FILE)
        
    print("Loading Data...")
    all_texts = []
    all_labels = []
    all_domains = []
    
    # LOAD AND LABEL
    # LOAD AND LABEL
    part1_files = [f for f in os.listdir(DATA_DIR_PART1) if f.endswith('.xlsx') and not f.startswith('~')]
    print(f"\nScanning {DATA_DIR_PART1} - Found {len(part1_files)} files.")

    for f in part1_files:
        fname = os.path.join(DATA_DIR_PART1, f)
        try:
            df = pd.read_excel(fname)
            # Detect cols
            txt_col = next((c for c in df.columns if 'text' in c.lower()), None)
            lbl_col = next((c for c in df.columns if 'label' in c.lower() or 'sentiment' in c.lower()), None)
            
            # Check for domain column
            dom_col = None
            if 'domain' in df.columns:
                 dom_col = 'domain'
            
            if txt_col and lbl_col:
                texts = df[txt_col].apply(clean_text).tolist()
                raw_labels = df[lbl_col].tolist()
                
                # Get domains directly from file, defaulting to Unknown if missing
                if dom_col:
                    file_domains = df[dom_col].fillna("Unknown").tolist()
                else:
                    file_domains = ["Unknown"] * len(texts)
                
                # Apply the robust mapper
                clean_labels = [map_sentiment_part1(l) for l in raw_labels]
                
                all_texts.extend(texts)
                all_labels.extend(clean_labels)
                all_domains.extend(file_domains)
                
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    # --- CRITICAL: DATA DEDUPLICATION LAYER ---
    # Part 1 datasets (specifically merged_dataset vs others) contain significant overlaps.
    # Without this step, Train/Test sets would share identical samples (Data Leakage),
    # leading to artificial 99% accuracy scores. Strict distinctness is enforced.
    if all_texts:
        df_labeled = pd.DataFrame({
            'text': all_texts, 
            'label': all_labels, 
            'domain': all_domains
        })
        initial_count = len(df_labeled)
        
        # Drop strict duplicates (same text content)
        df_labeled = df_labeled.drop_duplicates(subset=['text'], keep='first')
        
        all_texts = df_labeled['text'].tolist()
        all_labels = df_labeled['label'].tolist()
        all_domains = df_labeled['domain'].tolist()
        
        print(f"Deduplication Audit: Removed {initial_count - len(df_labeled)} duplicate rows to prevent leakage. Final Unique N={len(df_labeled)}")

    # --- DATA AUGMENTATION FOR CLASS BALANCE ---
    # The Neutral class (label=1) is significantly underrepresented.
    # We augment it to achieve better class balance.
    print("\nApplying Data Augmentation to Minority Class (Neutral)...")
    
    # Count class distribution
    label_counts = pd.Series(all_labels).value_counts().sort_index()
    print(f"Before Augmentation: {dict(label_counts)}")
    
    # Find the majority class size
    max_class_size = label_counts.max()
    
    # Augment Neutral class (label=1)
    neutral_indices = [i for i, l in enumerate(all_labels) if l == 1]
    neutral_texts = [all_texts[i] for i in neutral_indices]
    neutral_domains = [all_domains[i] for i in neutral_indices]
    
    # Calculate how many augmented samples we need
    # FIX: Target should be based on MAX of Negative and Positive classes
    target_neutral = int(max(label_counts.get(0, 0), label_counts.get(2, 0)) * 0.8)
    augment_count = max(0, target_neutral - len(neutral_texts))
    
    if augment_count > 0:
        print(f"Generating {augment_count} augmented Neutral samples...")
        augmented_texts = []
        augmented_labels = []
        augmented_domains = []
        
        for i in range(augment_count):
            # Sample from existing Neutral texts and augment
            idx = i % len(neutral_texts)
            aug_text = augment_text(neutral_texts[idx])
            augmented_texts.append(aug_text)
            augmented_labels.append(1)  # Neutral
            augmented_domains.append(neutral_domains[idx])
        
        # Add augmented samples to the dataset
        all_texts.extend(augmented_texts)
        all_labels.extend(augmented_labels)
        all_domains.extend(augmented_domains)
        
        label_counts_after = pd.Series(all_labels).value_counts().sort_index()
        print(f"After Augmentation: {dict(label_counts_after)}")


    # --- LOAD YOUTUBE DATA (UNLABELED) ---
    print("Loading YouTube comments (Part 2) for Embedding Enhancement...")
    yt_texts = []
    yt_domains = [] # For analysis later
    part2_root = "part2_data"
    if os.path.exists(part2_root):
        for root, dirs, files in os.walk(part2_root):
            for f in files:
                if f.endswith(".xlsx") and not f.startswith("~"):
                    try:
                        # Trying header=1 based on inspection
                        tmp = pd.read_excel(os.path.join(root, f), header=1)
                        if 'comment' in tmp.columns and 'domain' in tmp.columns:
                            clean_comments = tmp['comment'].dropna().astype(str).tolist()
                            doms = tmp['domain'].dropna().astype(str).tolist()
                            processed = [clean_text(c) for c in clean_comments]
                            yt_texts.extend(processed)
                            yt_domains.extend(doms)
                    except Exception:
                        pass
    print(f"Loaded {len(yt_texts)} unlabeled YouTube comments.")

    if not all_texts:
        print("No labeled data loaded! Exiting.")
        return

    # COMBINED CORPUS FOR EMBEDDINGS
    full_corpus = all_texts + yt_texts
    
    # TOKENIZATION
    print("Tokenizing...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(full_corpus) # Fit on EVERYTHING
    
    # Sequences for Labeled Data
    sequences = tokenizer.texts_to_sequences(all_texts)
    X = pad_sequences(sequences, maxlen=MAX_TOKENS)
    y = np.array(all_labels)
    domains = np.array(all_domains)
    vocab_size = len(tokenizer.word_index) + 1
    
    # Ensure directory exists
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")

    # LOAD EMBEDDINGS
    print("Loading Models...")
    
    # Word2Vec Logic
    w2v_path = "embeddings/word2vec.final.model"
    if os.path.exists(w2v_path):
        print("Loading existing Word2Vec model...")
        try:
            w2v_model = Word2Vec.load(w2v_path).wv
        except Exception as e:
            print(f"Error loading Word2Vec ({e}). Retraining...")
            tokenized_corpus = [t.split() for t in full_corpus]
            w2v_obj = Word2Vec(sentences=tokenized_corpus, vector_size=EMBEDDING_DIM, window=5, min_count=2, workers=12)
            w2v_obj.save(w2v_path)
            w2v_model = w2v_obj.wv
    else:
        print("Training Word2Vec on FULL CORPUS...")
        tokenized_corpus = [t.split() for t in full_corpus]
        w2v_obj = Word2Vec(sentences=tokenized_corpus, vector_size=EMBEDDING_DIM, window=5, min_count=2, workers=12)
        w2v_obj.save(w2v_path)
        w2v_model = w2v_obj.wv

    # FastText Logic
    ft_path = "embeddings/fasttext.final.model"
    if os.path.exists(ft_path):
        print("Loading existing FastText model...")
        try:
            ft_model = FastText.load(ft_path).wv
            print("Loaded FastText successfully.")
        except Exception as e:
            print(f"Error loading FastText ({e}). Retraining...")
            tokenized_corpus = [t.split() for t in full_corpus]
            ft_obj = FastText(sentences=tokenized_corpus, vector_size=EMBEDDING_DIM, window=5, min_count=2, workers=12)
            ft_obj.save(ft_path)
            ft_model = ft_obj.wv
    else:
        print("Training FastText on FULL CORPUS...")
        tokenized_corpus = [t.split() for t in full_corpus]
        ft_obj = FastText(sentences=tokenized_corpus, vector_size=EMBEDDING_DIM, window=5, min_count=2, workers=12)
        ft_obj.save(ft_path)
        ft_model = ft_obj.wv

    # BUILD MATRICES
    print("Building Embedding Matrices...")
    mat_w2v, oov_w2v = build_embedding_matrix(tokenizer.word_index, w2v_model, vocab_size)
    mat_ft, oov_ft = build_embedding_matrix(tokenizer.word_index, ft_model, vocab_size)
    
    print(f"Word2Vec OOV Rate: {oov_w2v:.4f}")
    print(f"FastText OOV Rate: {oov_ft:.4f}")
    save_text_report(f"OOV RATES:\nWord2Vec: {oov_w2v:.4f}\nFastText: {oov_ft:.4f}\n")
    
    # SPLIT DATA (Stratified by label to ensure class balance)
    # FIX: Create a dedicated validation set instead of using validation_split
    # This ensures the same validation data is used across all experiments
    X_train_full, X_test, y_train_full, y_test, d_train_full, d_test = train_test_split(
        X, y, domains, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Further split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=RANDOM_SEED, stratify=y_train_full
    )
    
    print(f"Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # --- EXPERIMENTS ---
    results = {}
    best_f1 = 0
    best_model = None
    
    # 1. W2V Frozen
    print("\nTraining W2V Frozen (with callbacks)...")
    # FIX: Create new callback instances for each experiment to avoid state issues
    callbacks = get_training_callbacks()
    model = build_gru_model(vocab_size, w2v_model.vector_size, MAX_TOKENS, mat_w2v, False)
    model.fit(X_train, y_train, epochs=15, batch_size=128, 
              validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    f1 = evaluate_model(model, X_test, y_test, "W2V Frozen")
    results['w2v_frozen'] = f1
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
    
    # 2. W2V Tuned
    print("\nTraining W2V Tuned (with callbacks)...")
    callbacks = get_training_callbacks()  # Fresh callbacks
    model = build_gru_model(vocab_size, w2v_model.vector_size, MAX_TOKENS, mat_w2v, True)
    model.fit(X_train, y_train, epochs=15, batch_size=128, 
              validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    f1 = evaluate_model(model, X_test, y_test, "W2V Tuned")
    results['w2v_tuned'] = f1
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
    
    # 3. FT Frozen
    print("\nTraining FT Frozen (with callbacks)...")
    callbacks = get_training_callbacks()  # Fresh callbacks
    model = build_gru_model(vocab_size, ft_model.vector_size, MAX_TOKENS, mat_ft, False)
    model.fit(X_train, y_train, epochs=15, batch_size=128, 
              validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    f1 = evaluate_model(model, X_test, y_test, "FT Frozen")
    results['ft_frozen'] = f1
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
    
    # 4. FT Tuned
    print("\nTraining FT Tuned (with callbacks)...")
    callbacks = get_training_callbacks()  # Fresh callbacks
    model = build_gru_model(vocab_size, ft_model.vector_size, MAX_TOKENS, mat_ft, True)
    model.fit(X_train, y_train, epochs=15, batch_size=128, 
              validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    f1 = evaluate_model(model, X_test, y_test, "FT Tuned")
    results['ft_tuned'] = f1
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
    
    # SAVE BEST MODEL
    print(f"\n--- Saving Best Model (F1={best_f1:.4f}) ---")
    best_model.save("best_sentiment_model.keras")
    print("Model saved to: best_sentiment_model.keras")
    
    # Plot Comparison
    plot_f1_comparison(results)

    # --- REQUIREMENT 6B: ANALYZE COLLECTED YOUTUBE DATA (Using Best Model, assume FT Tuned) ---
    print("\n--- Requirement 6B: Analyze Collected YouTube Data ---")
    if yt_texts:
        print(f"Predicting sentiments on {len(yt_texts)} collected comments...")
        yt_seqs = tokenizer.texts_to_sequences(yt_texts)
        X_yt = pad_sequences(yt_seqs, maxlen=MAX_TOKENS)
        yt_probs = model.predict(X_yt, verbose=0)
        yt_preds = yt_probs.argmax(axis=1)
        df_yt = pd.DataFrame({'domain': yt_domains, 'pred': yt_preds})
        plot_domain_predictions(df_yt)
        df_yt.to_excel("youtube_predictions.xlsx")
        
        print(f"{'Domain':<35} | {'Neg %':<6} | {'Neu %':<6} | {'Pos %':<6} | {'Count':<6}")
        print("-" * 75)
        for dom, group in df_yt.groupby('domain'):
            counts = group['pred'].value_counts(normalize=True).sort_index()
            neg = counts.get(0, 0.0) * 100
            neu = counts.get(1, 0.0) * 100
            pos = counts.get(2, 0.0) * 100
            print(f"{dom:<35} | {neg:5.1f}% | {neu:5.1f}% | {pos:5.1f}% | {len(group):<6}")

        save_text_report("\n--- TOP CONFIDENCE SAMPLES (YOUTUBE) ---")
        for label_idx, label_name in [(0, "Negative"), (2, "Positive")]:
            indices = np.where(yt_preds == label_idx)[0]
            if len(indices) > 0:
                probs = yt_probs[indices, label_idx]
                top_idx = indices[np.argsort(probs)[-5:][::-1]]
                save_text_report(f"\nTop 5 {label_name} Samples:")
                for idx in top_idx:
                    conf = yt_probs[idx, label_idx]
                    text = yt_texts[idx][:150].replace("\n", " ")
                    save_text_report(f"  [{conf:.4f}] {text}...")

    # --- DOMAIN-WISE EVALUATION (Using Best Model, assume FT Tuned) ---
    print("\n--- Domain-Wise Evaluation (FT Tuned) ---")
    save_text_report("\n--- DOMAIN WISE EVALUATION ---")
    unique_domains = np.unique(d_test)
    for dom in unique_domains:
        if dom == "Unknown": continue  # Skip Unknown for specific evaluation
        
        mask = (d_test == dom)
        n_dom = np.sum(mask)
        if n_dom == 0:
            print(f"[SKIP] Domain '{dom}' has 0 test samples — skipping evaluation.")
            continue
        if n_dom < 10:
            print(f"[WARN] Domain '{dom}' has only {n_dom} test samples. F1 will be high-variance.")
        evaluate_model(model, X_test[mask], y_test[mask], f"Domain: {dom}")

    print("\nAll tasks complete.")

    # --- EXPERIMENT B: DOMAIN SHIFT (Leave-One-Out) ---
    print("\n--- Experiment B: Domain Shift (Leave-One-Out) ---")
    save_text_report("\n--- DOMAIN SHIFT EVALUATION ---")
    
    unique_all_domains = np.unique(domains)
    valid_domains = [d for d in unique_all_domains if d != "Unknown"]
    
    for held_out_dom in valid_domains:
        print(f"\n[Shift Exp] Holding out: {held_out_dom}")
        
        # STRICT FILTER: Use only KNOWN domains for Shift Experiment
        # Train on (Known Domains - Held Out)
        # Test on (Held Out)
        
        # Mask: include only valid domains (no Unknown) AND exclude held_out
        train_mask = (domains != held_out_dom) & (domains != "Unknown")
        X_train_shift = X[train_mask]
        y_train_shift = y[train_mask]
        
        test_mask = (domains == held_out_dom)
        X_test_shift = X[test_mask]
        y_test_shift = y[test_mask]
        
        if len(X_test_shift) == 0:
            print("  Skipping (No samples).")
            continue
            
        # Re-build model from scratch
        model_shift = build_gru_model(vocab_size, ft_model.vector_size, MAX_TOKENS, mat_ft, True)
        
        # Train with consistent epochs — explicit val split, same strategy as main experiments.
        # Try stratified split first; fall back to plain split for tiny domain folds.
        callbacks = get_training_callbacks()
        try:
            X_tr_sh, X_val_sh, y_tr_sh, y_val_sh = train_test_split(
                X_train_shift, y_train_shift,
                test_size=0.1, random_state=RANDOM_SEED, stratify=y_train_shift
            )
        except ValueError:  # too few samples in some class for stratify
            X_tr_sh, X_val_sh, y_tr_sh, y_val_sh = train_test_split(
                X_train_shift, y_train_shift,
                test_size=0.1, random_state=RANDOM_SEED
            )
            print(f"  [WARN] stratify failed for fold '{held_out_dom}' — using plain split.")
        model_shift.fit(X_tr_sh, y_tr_sh, epochs=15, batch_size=128,
                        validation_data=(X_val_sh, y_val_sh),
                        callbacks=callbacks, verbose=1)
        
        # Evaluate
        evaluate_model(model_shift, X_test_shift, y_test_shift, f"Train: Others | Test: {held_out_dom}")

        # Memory Management: Explicitly clear the Keras session after each fold.
        # This prevents the computation graph from exploding during the loop.
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()
