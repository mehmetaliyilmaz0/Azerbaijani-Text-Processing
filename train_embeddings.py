# train_embeddings.py (REFACTORED for Subsampling)

import pandas as pd
from pathlib import Path
import logging
from gensim.models import Word2Vec, FastText

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_models(params, output_dir, sample_frac=1.0):
    """
    Trains models using a given parameter set on a fraction of the data.
    
    Args:
        params (dict): Hyperparameters.
        output_dir (Path): Directory to save models.
        sample_frac (float): Fraction of the data to use (e.g., 0.2 for 20%).
        
    Returns:
        tuple: (w2v_model_path, ft_model_path)
    """
    
    # --- 1. Load Sentences ---
    INPUT_DIR = Path("cleaned_data")
    files = [
        INPUT_DIR / "labeled-sentiment_2col.xlsx",
        INPUT_DIR / "test__1__2col.xlsx",
        INPUT_DIR / "train__3__2col.xlsx",
        INPUT_DIR / "train-00000-of-00001_2col.xlsx",
        INPUT_DIR / "merged_dataset_CSV__1__2col.xlsx",
    ]
    
    print(f"[{params['run_id']}] Loading sentences... (Sample Fraction: {sample_frac})")
    sentences = []
    for f in files:
        if not f.exists():
            print(f"WARNING: '{f}' file not found, skipping.")
            continue
        try:
            df = pd.read_excel(f, usecols=["cleaned_text"])
            
            # --- YENİ ADIM: VERİ ÖRNEKLEMESİ ---
            if sample_frac < 1.0:
                # Verinin tamamı yerine rastgele bir kısmını al.
                # random_state=42: Her denemenin AYNI %20'lik kısmı almasını
                # garanti eder, bu da deneyin 'adil' (reproducible) olmasını sağlar.
                df = df.sample(frac=sample_frac, random_state=42)
            
            sentences.extend(df["cleaned_text"].astype(str).str.split().tolist())
        except Exception as e:
            print(f"ERROR: Error while reading '{f}': {e}")
            
    if not sentences:
        print("CRITICAL ERROR: No sentences found for training. Is 'cleaned_data' folder empty?")
        return None, None

    print(f"[{params['run_id']}] Found {len(sentences)} total texts for this trial.")

    # --- 2. Train Word2Vec Model ---
    print(f"[{params['run_id']}] Training Word2Vec...")
    
    # --- KRİTİK AYAR: worker sayısını parametreden al ---
    workers = params.get('workers', 4) # Optuna'dan gelmezse varsayılan 4
    
    w2v_model = Word2Vec(
        sentences=sentences, 
        vector_size=params['vector_size'],
        window=params['window'],         
        min_count=params['min_count'],      
        sg=params['sg'],             
        negative=10,
        epochs=params['epochs'],        
        workers=workers # Değişkeni buraya ata
    )
    w2v_path = output_dir / "word2vec.model"
    w2v_model.save(str(w2v_path))
    print(f"[{params['run_id']}] Word2Vec model saved.")

    # --- 3. Train FastText Model ---
    print(f"[{params['run_id']}] Training FastText...")
    ft_model = FastText(
        sentences=sentences,
        vector_size=params['vector_size'],  
        window=params['window'],
        min_count=params['min_count'],
        sg=params['sg'],
        min_n=params.get('min_n', 3),
        max_n=params.get('max_n', 6),
        epochs=params['epochs'],        
        workers=workers # Değişkeni buraya ata
    )
    ft_path = output_dir / "fasttext.model"
    ft_model.save(str(ft_path))
    print(f"[{params['run_id']}] FastText model saved.")
    
    return w2v_path, ft_path

# --- SCRIPT'İ TEK BAŞINA ÇALIŞTIRMAK İÇİN (TEST AMAÇLI) ---
if __name__ == "__main__":
    print("--- Running 'train_embeddings.py' in standalone test mode ---")
    
    default_params = {
        'run_id': 'default_test_run', 'vector_size': 300,
        'window': 5, 'min_count': 3, 'sg': 1, 'epochs': 10, 'workers': 4
    }
    default_output_dir = Path("embeddings_default_test")
    default_output_dir.mkdir(exist_ok=True)
    
    # Test ederken %10'luk veri ile test et (sample_frac=0.1)
    train_models(default_params, default_output_dir, sample_frac=0.1)
    print("--- Standalone test run complete (on 10% data) ---")