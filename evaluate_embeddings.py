# evaluate_embeddings.py (REFACTORED)

import pandas as pd
from pathlib import Path
import logging
import re
from numpy import dot, float32 as REAL
from numpy.linalg import norm
from gensim.models import Word2Vec, FastText
import numpy as np

# Logging'i global alana taşı
logging.basicConfig(level=logging.ERROR)

# --- YARDIMCI FONKSİYONLAR (Değişmedi) ---
def lexical_coverage(model, tokens):
    vocab = model.wv.key_to_index
    return sum(1 for t in tokens if t in vocab) / max(1, len(tokens))

def read_tokens(f):
    df = pd.read_excel(f, usecols=["cleaned_text"])
    return [t for row in df["cleaned_text"].astype(str) for t in row.split()]

def cos_sim(a, b):
    return float(dot(a, b) / (norm(a) * norm(b)))

def pair_sim(model, pairs):
    vals = []
    for a, b in pairs:
        try:
            vec_a = model.wv[a]
            vec_b = model.wv[b]
            vals.append(cos_sim(vec_a, vec_b))
        except KeyError:
            pass
    return sum(vals) / max(1, len(vals)) if vals else float('nan')

def neighbors(model, word, k=5):
    try:
        return [w for w, score in model.wv.most_similar(word, topn=k)]
    except KeyError:
        return []
# ---

def evaluate_model_pair(w2v_path, ft_path):
    """
    Loads a pair of W2V and FT models and runs all evaluation tests.
    
    Args:
        w2v_path (Path): Path to the trained Word2Vec model.
        ft_path (Path): Path to the trained FastText model.
        
    Returns:
        dict: A dictionary containing all calculated scores
              (e.g., 'sep_w2v', 'sep_ft').
    """
    
    print(f"Loading models for evaluation: {w2v_path.parent.name}")
    
    if not w2v_path.exists() or not ft_path.exists():
        print(f"ERROR: Model files not found. '{w2v_path}' or '{ft_path}' is missing.")
        return None # Hata durumunda None döndür
        
    w2v = Word2Vec.load(str(w2v_path))
    ft = FastText.load(str(ft_path))
    print("Models loaded successfully.")

    # --- 1. Lexical Coverage (Sadece bilgilendirme amaçlı) ---
    print("== 1. Lexical Coverage ==")
    # ... (Bu bölüm, ana hedef metrik için kritik olmadığından kısa tutulabilir
    # veya loglama için eklenebilir, ancak şimdilik atlıyoruz.)
    
    # --- 2. Semantic Similarity Tests (ANA HEDEF) ---
    syn_pairs = [
        ("yaxşı", "əla"), ("bahalı", "qiymətli"), ("ucuz", "sərfəli"),
        ("pis", "bərbad"), ("gözəl", "qəşəng")
    ]
    ant_pairs = [
        ("yaxşı", "pis"), ("bahalı", "ucuz"), ("gözəl", "çirkin"),
        ("sevirəm", "nifrət")
    ]

    print("== 2. Calculating Semantic Similarity ==")
    syn_w2v = pair_sim(w2v, syn_pairs)
    syn_ft = pair_sim(ft, syn_pairs)
    ant_w2v = pair_sim(w2v, ant_pairs)
    ant_ft = pair_sim(ft, ant_pairs)
    
    sep_w2v = float('nan')
    sep_ft = float('nan')

    try:
        sep_w2v = syn_w2v - ant_w2v
        sep_ft = syn_ft - ant_ft
    except Exception:
        pass # 'nan' gelirse 'nan' olarak kalır

    # --- 3. Qualitative Test (Sadece ekrana basılır) ---
    print("== 3. Qualitative Nearest Neighbors ==")
    seed_words = ["yaxşı", "pis", "<PRICE>", "<RATING_POS>", "yox"]
    for word in seed_words:
        print(f"  '{word}' NN (W2V): {neighbors(w2v, word, k=3)}")
        print(f"  '{word}' NN (FT): {neighbors(ft, word, k=3)}")

    # --- 4. Return Results ---
    # Bu, 'run_experiments.py' script'inin alacağı sonuç sözlüğüdür.
    results = {
        'syn_w2v': syn_w2v,
        'ant_w2v': ant_w2v,
        'sep_w2v': sep_w2v,
        'syn_ft': syn_ft,
        'ant_ft': ant_ft,
        'sep_ft': sep_ft, # Bizim ana hedef metriğimiz
    }
    
    print(f"Evaluation complete. FastText Separation Score: {sep_ft:.4f}")
    return results

# --- SCRIPT'İ TEK BAŞINA ÇALIŞTIRMAK İÇİN (TEST AMAÇLI) ---
if __name__ == "__main__":
    """
    This block allows the script to be run directly for testing.
    It will try to evaluate models from a 'default_test_run' folder.
    """
    print("--- Running 'evaluate_embeddings.py' in standalone test mode ---")
    
    # 'train_embeddings.py' script'indeki varsayılan çıktı klasörü
    test_dir = Path("embeddings_default_test")
    
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        print("Please run 'train_embeddings.py' standalone first to create models.")
    else:
        test_w2v_path = test_dir / "word2vec.model"
        test_ft_path = test_dir / "fasttext.model"
        
        results = evaluate_model_pair(test_w2v_path, test_ft_path)
        if results:
            print("\n--- Standalone test results ---")
            print(results)