"""
CENG442 - NLP Assignment Part 2
Module: assign_domains.py
Description:
    This script assigns domain labels to Part 1 data (which lacks domain information)
    using keyword-based classification. It analyzes the text content and matches
    against domain-specific keywords to determine the most likely domain.
"""
import os
import pandas as pd
import re
from collections import Counter

# Domain keywords (from collect_youtube_data.py)
DOMAIN_KEYWORDS = {
    "Technology & Digital Services": [
        "telefon", "honor", "xiaomi", "samsung", "iphone", "internet", "tətbiq", "oyun", 
        "texnologiya", "kompüter", "playstation", "smart", "notebook", "macbook", "airpods",
        "bakcell", "azercell", "nar", "mobile", "app", "proqram", "sayt", "video", "youtube"
    ],
    "Finance & Business": [
        "biznes", "bank", "kredit", "investisiya", "manat", "dollar", "faiz", "sığorta",
        "sahibkar", "vergi", "maaş", "kripto", "bitcoin", "pul", "ödəniş", "kart", "ipoteka",
        "mühasib", "maliyyə", "qiymət", "baha", "ucuz", "endirim"
    ],
    "Social Life & Entertainment": [
        "mahnı", "meyxana", "toy", "film", "serial", "vlog", "komediya", "şou", "məşhur",
        "konsert", "klip", "vine", "tiktok", "əyləncə", "restoran", "kafe", "gəzinti",
        "səyahət", "bayram", "ad günü", "sevgi", "dostlar", "ailə"
    ],
    "Retail & Lifestyle": [
        "bazar", "mall", "mağaza", "geyim", "qızıl", "market", "ticarət", "alış-veriş",
        "paltar", "ayaqqabı", "aksesuar", "mebel", "ev", "mənzil", "maşın", "avtomobil",
        "zara", "lc waikiki", "brendlər", "stil", "moda", "gözəllik"
    ],
    "Public Services": [
        "asan", "nazirlik", "dövlət", "imtahan", "kommunal", "su", "işıq", "qaz", "pensiya",
        "sosial", "sığorta", "poliklinika", "xəstəxana", "nəqliyyat", "metro", "avtobus",
        "pasport", "şəxsiyyət", "qeydiyyat", "hökumət", "icra", "bələdiyyə"
    ]
}

def normalize_text(text):
    """Normalize text for keyword matching."""
    if not isinstance(text, str):
        return ""
    return text.lower().strip()

def classify_domain(text):
    """
    Classify text into a domain based on keyword matching.
    Returns the domain with the most keyword matches, or 'Unknown' if no matches.
    """
    text = normalize_text(text)
    if not text:
        return "Unknown"
    
    scores = Counter()
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            # \b word-boundary prevents substring false-positives (e.g. "pul" matching "pulluk")
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text):
                scores[domain] += 1
    
    if not scores:
        return "Unknown"
    
    # Return domain with highest score
    return scores.most_common(1)[0][0]

def process_file(filepath, output_dir):
    """Process a single Part 1 file and add domain column."""
    filename = os.path.basename(filepath)
    print(f"\nProcessing: {filename}")
    
    df = pd.read_excel(filepath)
    print(f"  Original shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Find text column
    text_col = None
    for col in df.columns:
        if 'text' in col.lower():
            text_col = col
            break
    
    if not text_col:
        print(f"  ERROR: No text column found!")
        return None
    
    # Classify each row
    print("  Classifying domains...")
    df['domain'] = df[text_col].apply(classify_domain)
    
    # Show distribution
    domain_counts = df['domain'].value_counts()
    print(f"\n  Domain Distribution:")
    for domain, count in domain_counts.items():
        pct = count / len(df) * 100
        print(f"    {domain}: {count} ({pct:.1f}%)")
    
    # Save to output directory
    output_path = os.path.join(output_dir, f"domain_{filename}")
    df.to_excel(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    return df

def main():
    # Part 1 files to process
    part1_files = [
        "train__3_.xlsx",
        "test__1_.xlsx", 
        "train-00000-of-00001.xlsx",
        "merged_dataset_CSV__1_.xlsx",
        "labeled-sentiment.xlsx"
    ]
    
    output_dir = "part1_with_domains"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 60)
    print("DOMAIN ASSIGNMENT FOR PART 1 DATA")
    print("=" * 60)
    
    total_stats = Counter()
    
    for filepath in part1_files:
        if os.path.exists(filepath):
            df = process_file(filepath, output_dir)
            if df is not None:
                for domain, count in df['domain'].value_counts().items():
                    total_stats[domain] += count
        else:
            print(f"\nFile not found: {filepath}")
    
    print("\n" + "=" * 60)
    print("TOTAL DOMAIN DISTRIBUTION (All Files)")
    print("=" * 60)
    total = sum(total_stats.values())
    for domain, count in total_stats.most_common():
        pct = count / total * 100
        print(f"  {domain}: {count} ({pct:.1f}%)")
    print(f"\n  TOTAL: {total} samples")

if __name__ == "__main__":
    main()
