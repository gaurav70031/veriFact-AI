# Datasets — Fake News Detection

This document lists every dataset supported by the preprocessing pipeline,
their sources, licences, and any usage restrictions.

**No dataset files are included in this repository.**
You must download them separately and place them in `ml/datasets/raw/`.

---

## 1. ISOT Fake News Dataset (Primary — Recommended)

| Field            | Value |
|------------------|-------|
| **Name**         | ISOT Fake News Dataset |
| **Creator**      | University of Victoria — Information Security and Object Technology (ISOT) Lab |
| **Files**        | `Fake.csv`, `True.csv` |
| **Size**         | ~44,898 articles (23,481 fake + 21,417 real) |
| **Language**     | English |
| **Domain**       | Political news (2015–2018) |
| **Format**       | CSV — columns: `title`, `text`, `subject`, `date` |
| **Labels**       | Implicit via file (Fake.csv = 0, True.csv = 1) |

### Source

- **Kaggle mirror (easiest):**
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- **Original paper:**
  Ahmed H, Traore I, Saad S. "Detecting opinion spams and fake news using text classification."
  *Journal of Security and Privacy*, 2018.
  https://doi.org/10.1002/spy2.9

### Licence / Usage Restrictions

The ISOT dataset is made available for **academic and research purposes only**.

- ✅ Permitted: academic research, student projects, non-commercial experimentation.
- ❌ Not permitted: commercial use, redistribution without permission.
- The dataset page on Kaggle requires accepting the dataset's terms of use before
  downloading.  Review those terms before use.
- Real articles in `True.csv` were collected from Reuters.com.
  Reuters content is subject to Reuters copyright.
  **Do not redistribute the raw text** — use only for local model training.
- Fake articles in `Fake.csv` were collected from PolitiFact and other sources
  flagged during the 2016 US election period.

### How to Download

**Option A — Kaggle API (automated):**
```bash
# Requires ~/.kaggle/kaggle.json with your API credentials
# See: https://www.kaggle.com/docs/api
pip install kaggle
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p ml/datasets/raw/ --unzip
```

**Option B — Manual:**
1. Visit https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Sign in and accept the dataset terms.
3. Download and unzip.
4. Place `Fake.csv` and `True.csv` in `ml/datasets/raw/`.

### Expected File Layout After Download

```
ml/datasets/raw/
├── Fake.csv    (~62 MB)
└── True.csv    (~54 MB)
```

---

## 2. LIAR Dataset (Optional — Multi-class)

| Field            | Value |
|------------------|-------|
| **Name**         | LIAR: A Benchmark Dataset for Fake News Detection |
| **Creator**      | William Yang Wang, UC Santa Barbara |
| **Files**        | `train.tsv`, `valid.tsv`, `test.tsv` |
| **Size**         | ~12,836 short statements |
| **Language**     | English |
| **Domain**       | Political statements from PolitiFact (2007–2016) |
| **Format**       | TSV — 14 columns (no header), see LIAR paper |
| **Labels**       | 6-class: true / mostly-true / half-true / barely-true / false / pants-fire |
| **Binary mapping** | true / mostly-true / half-true → REAL; barely-true / false / pants-fire → FAKE |

### Source

- **HuggingFace Datasets:**
  https://huggingface.co/datasets/liar
- **Original paper:**
  Wang W Y. "'Liar, Liar Pants on Fire': A New Benchmark Dataset for Fake News Detection."
  *ACL 2017*. https://aclanthology.org/P17-2067/
- **Direct download:**
  https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

### Licence / Usage Restrictions

- The LIAR dataset is released for **research purposes**.
- Statements were sourced from PolitiFact — no commercial redistribution.
- ✅ Permitted: academic research, benchmarking, student projects.
- ❌ Not permitted: commercial use without separate permission from PolitiFact.

### How to Download

```bash
# Via HuggingFace datasets library
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('liar')
ds['train'].to_csv('ml/datasets/raw/liar_train.tsv', sep='\t', index=False)
ds['validation'].to_csv('ml/datasets/raw/liar_valid.tsv', sep='\t', index=False)
ds['test'].to_csv('ml/datasets/raw/liar_test.tsv', sep='\t', index=False)
"
```

### Notes

- LIAR contains short political claims (~20 words), unlike ISOT which has full articles.
- Models trained on ISOT may not transfer well to LIAR and vice versa.
- The preprocessing pipeline's binary mapping (`half-true` → REAL) is a simplification.
  Treat LIAR results as supplementary, not primary evaluation.

---

## 3. Generic / Custom Datasets

The pipeline supports any CSV, TSV, JSON, or JSONL file that contains
at minimum a **text column** and a **label column**.

### Supported label values (auto-detected)

| Your label string | Mapped to |
|-------------------|-----------|
| `fake`, `FAKE`, `0`, `false`, `misinformation` | 0 (FAKE) |
| `real`, `REAL`, `1`, `true`, `reliable` | 1 (REAL) |

See `ml/preprocessing/config.py` — `FAKE_LABEL_ALIASES` / `REAL_LABEL_ALIASES`
to add more aliases without modifying source code.

### Column auto-detection

The loader searches for these column name patterns:

| Type  | Searched patterns |
|-------|-------------------|
| Text  | `text`, `body`, `content`, `article`, `statement` |
| Title | `title`, `headline`, `subject`, `head` |
| Label | `label`, `class`, `target`, `fake`, `real`, `category` |

Use `--inspect` to check what columns a file will resolve to before running:
```bash
python scripts/prepare_dataset.py --inspect ml/datasets/raw/yourfile.csv
```

---

## Data Placement Summary

```
ml/datasets/raw/
├── Fake.csv          ← ISOT fake articles     (download separately)
├── True.csv          ← ISOT real articles      (download separately)
├── liar_train.tsv    ← LIAR train split        (optional, download separately)
├── liar_valid.tsv    ← LIAR validation split   (optional)
├── liar_test.tsv     ← LIAR test split         (optional)
└── <your_file>.csv   ← Custom dataset          (optional)

ml/datasets/processed/   ← Generated by scripts/prepare_dataset.py
├── train.csv
├── val.csv
├── test.csv
└── dataset_stats.json
```

---

## Running the Pipeline

```bash
# ISOT (primary)
python scripts/prepare_dataset.py \
    --files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv

# ISOT with transformer-mode cleaning
python scripts/prepare_dataset.py \
    --files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv \
    --mode transformer

# LIAR
python scripts/prepare_dataset.py \
    --files ml/datasets/raw/liar_train.tsv \
            ml/datasets/raw/liar_valid.tsv \
            ml/datasets/raw/liar_test.tsv

# Inspect a file before processing
python scripts/prepare_dataset.py --inspect ml/datasets/raw/Fake.csv
```

---

## Copyright Notice

This project does **not** include, redistribute, or reproduce any dataset
content.  All dataset files must be obtained directly from the authoritative
sources listed above.  Users are responsible for complying with the terms
and conditions of each dataset they use.
