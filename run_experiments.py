# run_experiments.py (Paralel + Subsampling Sürümü)

import pandas as pd
from pathlib import Path
import time
import shutil
import numpy as np
import optuna
import os # CPU sayısını almak için

# Refactor ettiğimiz fonksiyonları import et
from train_embeddings import train_models
from evaluate_embeddings import evaluate_model_pair

# =================================================================
# DENEY KONFİGÜRASYONU (BURAYI AYARLA)
# =================================================================

# --- 1. Hızlandırma Ayarları ---

# Verinin ne kadarlık bir kısmını kullanalım? (örn: 0.2 = %20)
# 1.0 = Verinin tamamı (Yavaş)
# 0.2 = Hızlı keşif (Tavsiye edilen)
SAMPLE_FRAC = 0.37 

# Toplam kaç deneme (trial) yapılsın?
N_TRIALS = 50

# --- 2. Kaynak (CPU) Yönetimi ---
# 
# KRİTİK KAYNAK UYARISI:
# Toplam CPU kullanımı = N_JOBS * GENSIM_WORKERS
# Eğer 8 çekirdekli bir CPU'nuz varsa:
#   - Strateji 1: N_JOBS=4, GENSIM_WORKERS=2 (Tavsiye edilen)
#   - Strateji 2: N_JOBS=2, GENSIM_WORKERS=4
#   - Strateji 3: N_JOBS=8, GENSIM_WORKERS=1
# N_JOBS * GENSIM_WORKERS, toplam fiziksel çekirdek sayınızı aşarsa,
# sistem "boğulur" (CPU thrashing) ve yavaşlar.

# Optuna'nın aynı anda çalıştıracağı paralel işlem sayısı
N_JOBS = 4 

# Her bir Optuna işleminin, gensim eğitimi için kullanacağı çekirdek sayısı
GENSIM_WORKERS = 2 

print(f"--- FİZİKSEL ÇEKİRDEK SAYISI (TAHMİNİ): {os.cpu_count()} ---")
print(f"--- OPTUNA AYARLARI ---")
print(f"Aynı anda {N_JOBS} deneme çalışacak.")
print(f"Her deneme {GENSIM_WORKERS} gensim çekirdeği kullanacak.")
print(f"Tahmini Toplam Çekirdek Kullanımı: {N_JOBS * GENSIM_WORKERS}")
print(f"Veri Örneklemi: {SAMPLE_FRAC * 100}%")
print(f"Toplam Deneme: {N_TRIALS}")
print("--------------------------")
time.sleep(3) # Kullanıcının ayarları görmesi için 3 saniye bekle

# --- 3. Optuna Depolama (Storage) ---
# Paralel çalıştırma (N_JOBS > 1) için bu ZORUNLUDUR.
# Farklı işlemlerin hangi denemeyi yapacaklarını koordine etmesini sağlar.
STUDY_NAME = "ceng442-opt-v2" # v1'den ayırmak için
STORAGE_URL = "sqlite:///optuna_study.db" # SQLite veritabanı dosyası

# --- 4. Hedef Metrik ---
TARGET_METRIC = 'sep_ft' 

# Deney dosyalarının kaydedileceği yer
EXPERIMENTS_DIR = Path("optuna_experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# =================================================================
# OPTUNA HEDEF FONKSİYONU (OBJECTIVE FUNCTION)
# =================================================================

def objective(trial):
    """
    Optuna'nın her denemede (paralel olarak) çalıştıracağı ana fonksiyon.
    """
    
    # --- 1. Parametre Arama Alanını (Search Space) Tanımla ---
    params = {
        'run_id': f"trial_{trial.number}",
        'vector_size': trial.suggest_categorical('vector_size', [100, 200, 300, 400, 500]),
        'sg': 1,
        'workers': GENSIM_WORKERS, # Kaynak ayarını parametreye ekle
        'min_count': trial.suggest_int('min_count', 1, 3),
        'window': trial.suggest_int('window', 2, 10),
        'epochs': trial.suggest_int('epochs', 5, 20), # Üst sınırı 40'a çıkarabiliriz
    }

    print(f"\n[DENEME {trial.number}] BAŞLIYOR. Parametreler: {params}")
    
    run_output_dir = EXPERIMENTS_DIR / params['run_id']
    run_output_dir.mkdir(exist_ok=True)
    
    try:
        # --- 2. Modeli Eğit (Subsampling ile) ---
        w2v_path, ft_path = train_models(params, run_output_dir, sample_frac=SAMPLE_FRAC)
        
        if w2v_path is None:
            print(f"[Deneme {trial.number}] Eğitim başarısız oldu. Budanıyor (Pruned).")
            raise optuna.exceptions.TrialPruned()
            
        # --- 3. Modeli Değerlendir ---
        eval_results = evaluate_model_pair(w2v_path, ft_path)
        
        if eval_results is None:
            print(f"[Deneme {trial.number}] Değerlendirme başarısız oldu. Budanıyor (Pruned).")
            raise optuna.exceptions.TrialPruned()

        # --- 4. Disk Temizliği ---
        # Hızlandırma için bu adımı geçici olarak kapatabiliriz,
        # ancak diskte hızla yer (GB'larca) dolacaktır.
        # try:
        #     shutil.rmtree(run_output_dir)
        #     print(f"[Deneme {trial.number}] Model dosyaları temizlendi.")
        # except OSError as e:
        #     print(f"UYARI: {run_output_dir.name} temizlenemedi: {e}")

        # --- 5. Hedef Skoru Döndür ---
        score = eval_results.get(TARGET_METRIC)
        
        if score is None or np.isnan(score):
            print(f"[Deneme {trial.number}] 'nan' skor döndürdü. Budanıyor (Pruned).")
            raise optuna.exceptions.TrialPruned()
        
        print(f"[DENEME {trial.number}] TAMAMLANDI. Skor ({TARGET_METRIC}): {score:.6f}")
        return score

    except Exception as e:
        print(f"[DENEME {trial.number}] KRİTİK HATA ile çöktü: {e}")
        return -1.0 # Başarısız denemeyi -1.0 ile cezalandır

# =================================================================
# OPTUNA ÇALIŞTIRMA BLOĞU
# =================================================================
if __name__ == "__main__":
    
    print("--- Optuna Hiperparametre Optimizasyonu Başlatılıyor ---")
    
    # Paralel çalıştırma için 'storage' ve 'load_if_exists' ZORUNLUDUR.
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        load_if_exists=True,
        direction='maximize'
    )
    
    # Optimizasyonu başlat
    # n_jobs=N_JOBS -> Paralel çalıştırmayı aktive eder
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu. O ana kadarki sonuçlar kaydediliyor.")
    
    # --- Sonuçları Kaydet ve Göster ---
    print(f"\n{'='*50}")
    print("TÜM DENEYLER TAMAMLANDI.")
    
    # Veritabanındaki tüm denemelerin sonuçlarını al
    df = study.trials_dataframe()
    df = df.dropna(subset=['value'])
    df.to_csv("optuna_results_parallel.csv", index=False)
    print(f"Tüm sonuçlar 'optuna_results_parallel.csv' dosyasına kaydedildi.")
    
    print("\n--- En İyi Sonuç ---")
    try:
        print(f"En Yüksek Skor ({TARGET_METRIC}): {study.best_value:.6f}")
        print("En İyi Parametreler:")
        print(study.best_params)
    except optuna.exceptions.OptunaError:
        print("Hiçbir başarılı deneme tamamlanamadı.")