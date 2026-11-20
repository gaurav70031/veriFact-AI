"""
download_nltk_data.py
=====================
Download the NLTK data packages required by the VeriFact AI preprocessing
pipeline (lemmatization and stopword removal).

Required packages
-----------------
  punkt                     - Sentence tokenizer (Punkt)
  punkt_tab                 - Punkt tokenizer tables
  averaged_perceptron_tagger - POS tagger (required for lemmatization)
  averaged_perceptron_tagger_eng - English POS tagger tables
  wordnet                   - WordNet lexical database (for lemmatization)
  omw-1.4                   - Open Multilingual Wordnet
  stopwords                 - English stopword list

Usage
-----
    python scripts/download_nltk_data.py

Data is saved to ~/nltk_data (default NLTK download directory).
Run once per machine or container.
"""

import sys

try:
    import nltk
except ImportError:
    print("[ERROR] NLTK not installed. Run: pip install nltk")
    sys.exit(1)

PACKAGES = [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "omw-1.4",
    "stopwords",
]

print("Downloading NLTK data packages...")
failed = []
for pkg in PACKAGES:
    try:
        result = nltk.download(pkg, quiet=False)
        if result:
            print(f"  OK  {pkg}")
        else:
            print(f"  --  {pkg} (already up to date or failed silently)")
    except Exception as exc:
        print(f"  FAILED  {pkg}: {exc}")
        failed.append(pkg)

print()
if failed:
    print(f"[WARN] {len(failed)} package(s) failed to download: {failed}")
    print("Lemmatization will fall back to simple whitespace tokenization.")
else:
    print("All NLTK data packages downloaded successfully.")
    print()
    print("Verify with:")
    print("  python -c \"from ml.preprocessing.lemmatizer import lemmatize_text; print(lemmatize_text('The cats are running quickly'))\"")
