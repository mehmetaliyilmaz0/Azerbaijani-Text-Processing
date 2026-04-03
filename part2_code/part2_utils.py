"""
Azerbaijani Sentiment Analysis Pipeline
Module: part2_utils.py
Description: 
    This module implements a specialized language filtering logic to distinguish 
    Azerbaijani text from Turkish and other dialects. It employs a two-layer 
    heuristic approach:
    1. Orthographic Layer: Detecting unique characters (e.g., 'ə').
    2. Lexical Layer: Analyzing distinctive markers and suffix patterns.
    
    This ensures that the collected YouTube dataset is strictly Azerbaijani, 
    preventing cross-lingual noise in the sentiment analysis pipeline.
"""
import re

# =================================================================
# DOMAIN CONFIGURATION
# =================================================================
# These domains form the structural backbone of the dataset.
# Mapped verbatim from the project requirements to ensure compliance.
DOMAINS = [
    "Technology & Digital Services",
    "Finance & Business",
    "Social Life & Entertainment",
    "Retail & Lifestyle",
    "Public Services"
]

# =================================================================
# AZERBAIJANI FILTER LOGIC (ADVANCED)
# =================================================================

AZ_STRONG_CHARS = set("əƏ")

# --- DISTINCTIVE LEXICONS ---
# Words that strongly suggest Azerbaijani and are NOT common in Turkish.
# These served as positive signals in the scoring algorithm.
AZ_MARKERS = {
    # Pronouns (Distinct from Turkish 'ben', 'kendi')
    "mən", "sən", "özüm", "özün", "özü", 
    # Conjunctions / Adverbs
    "ilə", "amma", "ancaq", "çünki", "əgər", "həmçinin", "gərək", 
    "xeyr", "bəli", "hə", "yox", "hələ", "yenə", "artıq", "bəlkə",
    "haçan", "harda", "burda", "orda", 
    # Functional & High-Frequency Words
    "üçün", "kimi", "görə", "sarı", "yana", "qarşı", "qədər",
    "haqqında", "barəsində", "savayı",
    # Interrogatives
    "niyə", "necə", "hansı", "nə",
    # Domain-Agnostic Vocabulary
    "yaxşı", "pis", "uşaq", "pul", "məktəb", "müəllim", "kitab",
    "tələbə", "universitet", "dövlət", "respublika", "rayon", "kənd",
    "şəhər", "küçə", "maşın", "vəziyyət", "sual", "cavab",
    "oldu", "olur", "edir", "etmir", "gəlir", "gedir", "baxır", 
    "danış", "oxu", "işlə", "istə", "qoy", "götür"
}

# Words that strongly suggest Turkish and are NOT common in Azerbaijani
TR_MARKERS = {
    # Pronouns
    "ben", "kendi", "kendim", "kendin",
    # Conjunctions / Adverbs
    "ve", "ile", "fakat", "çünkü", "eğer", "ayrıca", "belki", "keşke",
    "hayır", "evet", "yok", "henüz", "yine", "artık", 
    "nerede", "burada", "orada",
    # Function words
    "için", "gibi", "kadar", "hakkında", 
    # Question words
    "neden", "niçin", "nasıl", "hangi", "ne zaman",
    # Common specific words
    "güzel", "kötü", "çocuk", "para", "okul", "öğretmen", "kitap",
    "öğrenci", "üniversite", "devlet", "ilçe", "köy",
    "araba", "durum", "soru", "cevap",
    "oldu", "oluyor", "yapıyor", "yapmıyor", "geliyor", "gidiyor", "bakıyor",
    "konuş", "oku", "çalış", "iste", "koy", "al",
    # Slang / Fillers
    "şey", "bence", "sence", "abi", "kanka", "lan", "valla", "billaha",
    "gerçekten", "tabii", "aynen", "kusura", "bakma"
}

# --- SUFFIX HEURISTICS ---
# Turkish Present Continuous "-yor" is extremely distinctive.
# Azerbaijani uses "-ir", "-ır", "-ur", "-ür" (e.g., gəlir vs geliyor).
TR_SUFFIXES = [
    "yor", "yorsun", "yoruz", "yorsunuz", "yorlar", 
    "miyor", "mıyor", "meyor", "mayor" # Negations (checking end of word)
]

def normalize_text_for_filter(t: str) -> str:
    if not isinstance(t, str): return ""
    t = t.lower().strip()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t

def az_likelihood_score(text: str) -> float:
    t = normalize_text_for_filter(text)
    
    score = 0.0

    # 1. Strong Character Signal - 'ə' is unique to Azerbaijani
    if any(ch in t for ch in "əƏ"):
        score += 4.0

    # 2. Secondary Character Signals
    # q and x are very common in Az, rare in Tr (k/h usually)
    qx_count = t.count("q") + t.count("x")
    if qx_count >= 2:
        score += 1.5

    # 3. Token-based Analysis
    tokens = re.findall(r"[a-zəöüğışç]+", t)
    token_set = set(tokens)

    # 3a. Positive Matches
    az_hits = len(token_set.intersection(AZ_MARKERS))
    score += (az_hits * 1.2)

    # 3b. Negative Matches (Penalty)
    tr_hits = len(token_set.intersection(TR_MARKERS))
    score -= (tr_hits * 2.0) # Higher penalty for stronger separation

    # 4. Suffix Analysis (Heuristic for Turkish 'yor')
    # Check words longer than 4 chars ending in -yor variations
    tr_suffix_count = 0
    for tok in tokens:
        if len(tok) > 4:
            for suf in TR_SUFFIXES:
                if tok.endswith(suf):
                    tr_suffix_count += 1
                    break 
    
    if tr_suffix_count > 0:
        score -= (tr_suffix_count * 2.5) # Heavy penalty for -yor

    return score

def is_azerbaijani(text: str, threshold: float = 2.0) -> bool:
    """
    Determines if text is Azerbaijani.
    Adjusted default threshold to 2.0 after improved scoring.
    """
    return az_likelihood_score(text) >= threshold

# --- TEST BLOCK (verification logic implied) ---
if __name__ == "__main__":
    # Simple manual test to verify logic
    test_sentences = [
        ("Mən bu kitabı çox bəyəndim, əla idi.", True),
        ("Ben bu kitabı çok beğendim, güzeldi.", False),
        ("Salam, necəsən? İşlər necə gedir?", True),
        ("Selam, nasılsın? İşler nasıl gidiyor?", False),
        ("Hər şey yaxşıdır, narahat olma.", True),
        ("Her şey yolunda, merak etme.", False),
        ("Bu video çok güzel olmuş abi.", False), # Turkish slang
        ("Bu video çox gözəl olub qardaş.", True),
        ("Məncə bu tətbiq yaxşı işləyir.", True),
        ("Bence bu uygulama iyi çalışıyor.", False)
    ]
    
    print("Running Filter Diagnostic...")
    print(f"{'Sentence':<50} | {'Score':<6} | {'Pred':<5} | {'Target':<5}")
    print("-" * 75)
    for sent, target in test_sentences:
        sc = az_likelihood_score(sent)
        pred = is_azerbaijani(sent)
        status = "OK" if pred == target else "FAIL"
        print(f"{sent[:47]:<50} | {sc:5.1f}  | {str(pred):<5} | {str(target):<5} [{status}]")
