# IR Labs – Information Retrieval System

A comprehensive Python implementation of classical and modern information retrieval models with evaluation metrics and visualization. This project demonstrates core IR concepts including indexing, retrieval algorithms, and performance evaluation on the MEDLINE biomedical document collection.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Retrieval Models](#retrieval-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Interactive UI](#interactive-ui)
- [Configuration](#configuration)

## Features

✨ **Multiple Retrieval Models**

- Vector Space Model (VSM) with Cosine Similarity
- BM25 (Okapi BM25)
- Binary Independence Retrieval (BIR)
- BIR with Relevance Information
- Extended BIR variants
- Latent Semantic Indexing (LSI)
- Language Models (MLE, Laplace, Jelinek-Mercer, Dirichlet smoothing)

📊 **Comprehensive Evaluation Metrics**

- Precision & Recall
- Mean Average Precision (MAP)
- Interpolated MAP (IMAP)
- Precision@K
- R-Precision
- Reciprocal Rank
- DCG & NDCG
- F1-Score

🔄 **Text Processing Pipeline**

- Tokenization with regex patterns
- Stemming (Porter stemmer)
- Stopword removal
- MEDLINE document parsing

📈 **Visualization & Analysis**

- Interactive Streamlit UI
- PR curves for model comparison
- Gain charts
- Scalar metrics dashboard

💾 **Efficient Indexing**

- Inverted index construction
- TF-IDF matrix caching
- Index serialization (pkl format)

## Project Structure

```
RI/
├── main.py                          # Entry point for batch evaluation
├── config.py                        # Centralized configuration
├── README.md                        # This file
│
├── data/                            # MEDLINE collection
│   ├── MED.ALL                      # Documents (~1.33M documents)
│   ├── MED.QRY                      # Queries
│   ├── MED.REL                      # Relevance judgments
│   └── MED.REL.OLD                  # Legacy relevance data
│
├── preprocessing/                   # Text processing
│   ├── indexer.py                   # Index building & querying
│   ├── parser.py                    # MEDLINE format parser
│   ├── tokenizer.py                 # Token extraction
│   ├── stemmer.py                   # Stemming
│   └── stopwords.py                 # Stopword management
│
├── retrieval_models/                # IR algorithms
│   ├── run_model.py                 # Model dispatcher
│   ├── vsm/
│   │   └── cosinsim.py              # Cosine similarity
│   ├── BM25/
│   │   └── bm25.py                  # BM25 ranking
│   ├── BIR/
│   │   ├── bir.py                   # Standard BIR
│   │   ├── bir_rel.py               # BIR with relevance
│   │   ├── ex_bir.py                # Extended BIR
│   │   └── ex_bir_rel.py            # Extended BIR with relevance
│   ├── LSI/
│   │   └── lsi.py                   # Latent Semantic Indexing
│   └── lm/
│       ├── mle.py                   # Maximum Likelihood Estimation
│       ├── laplace.py               # Laplace smoothing
│       ├── jm.py                    # Jelinek-Mercer smoothing
│       └── dirichlet.py             # Dirichlet smoothing
│
├── Evaluation/                      # Metrics
│   ├── precision.py
│   ├── recall.py
│   ├── f1_score.py
│   ├── map.py
│   ├── imap.py
│   ├── precision_a_k.py
│   ├── r_precision.py
│   ├── reciprocal_rank.py
│   ├── dcg.py
│   ├── ndcg.py
│   └── gain.py
│
├── UI/                              # Streamlit interface
│   ├── app.py                       # Main UI application
│   ├── data_loader.py               # Results loading
│   ├── scalar_metrics.py            # Metrics display
│   ├── pr_curves.py                 # PR curve visualization
│   ├── gain.py                      # Gain chart display
│   ├── compute_results.py           # Result computation
│   └── results.json                 # Cached results
│
└── results/                         # Output directory
    ├── index_cache/                 # Serialized indices & matrices
    ├── plots/                       # Generated visualizations
    └── rankings/                    # Query rankings
```

## Installation

### Prerequisites

- Python 3.11+
- Conda (recommended)

### Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd RI
   ```

2. **Create the Conda environment:**

   ```bash
   conda env create -f environment.yml
   conda activate ir_lab5
   ```

3. **Verify installation:**
   ```bash
   python -c "import nltk; print('✓ NLTK installed')"
   ```

### Environment

The project requires:

- **Python 3.11**
- **NLTK** ≥3.8 (NLP toolkit)
- **NumPy** ≥1.26 (Numerical computing)
- **SciPy** ≥1.12 (Scientific computing)
- **scikit-learn** ≥1.4 (ML library, for LSI & metrics)
- **Pandas** ≥2.2 (Data manipulation)
- **Matplotlib** ≥3.8 (Visualization)
- **Streamlit** (Interactive UI)

See `environment.yml` for exact versions.

## Quick Start

### 1. **Prepare MEDLINE Data**

Ensure the data files are present in the `data/` directory:

- `MED.ALL` – Full document collection
- `MED.QRY` – Query set
- `MED.REL` – Relevance judgments

### 2. **Run Batch Evaluation**

Execute the main script to evaluate all retrieval models:

```bash
python main.py
```

This will:

1. Build or load a cached inverted index
2. Load all queries and relevance judgments
3. Run each query through all 11 retrieval models
4. Generate rankings and evaluation metrics
5. Save results to `results/rankings/`

### 3. **Launch Interactive UI**

View results with the Streamlit dashboard:

```bash
streamlit run UI/app.py
```

Open your browser to `http://localhost:8501`

## Retrieval Models

### Vector Space Model (VSM)

- **Algorithm:** Cosine similarity between TF-IDF vectors
- **Use case:** Baseline dense retrieval
- **File:** `retrieval_models/vsm/cosinsim.py`

### BM25 (Okapi BM25)

- **Algorithm:** Probabilistic ranking with saturation functions
- **Parameters:** `k1=1.5`, `b=0.75` (configurable)
- **Use case:** Industry standard, robust baseline
- **File:** `retrieval_models/BM25/bm25.py`

### Binary Independence Retrieval (BIR)

- **Variants:**
  - `bir.py` – Standard BIR without relevance
  - `bir_rel.py` – BIR with relevance judgments
  - `ex_bir.py` – Extended BIR
  - `ex_bir_rel.py` – Extended BIR with relevance
- **File:** `retrieval_models/BIR/`

### Language Models

- **MLE** – Maximum Likelihood Estimation (unsmoothed)
- **Laplace** – Add-one smoothing
- **Jelinek-Mercer** – Linear interpolation with collection model
- **Dirichlet** – Dirichlet smoothing (recommended)
- **Parameter:** `λ` or `μ` (configurable per model)
- **File:** `retrieval_models/lm/`

### Latent Semantic Indexing (LSI)

- **Algorithm:** Dimensionality reduction via SVD
- **Parameter:** `k` (number of dimensions, default: 300)
- **Use case:** Semantic retrieval, handles synonymy
- **File:** `retrieval_models/LSI/lsi.py`

## Evaluation Metrics

| Metric          | Description                                  | File                            |
| --------------- | -------------------------------------------- | ------------------------------- |
| **Precision**   | Fraction of retrieved docs that are relevant | `Evaluation/precision.py`       |
| **Recall**      | Fraction of relevant docs that are retrieved | `Evaluation/recall.py`          |
| **F1-Score**    | Harmonic mean of precision & recall          | `Evaluation/f1_score.py`        |
| **MAP**         | Average precision across all queries         | `Evaluation/map.py`             |
| **IMAP**        | Interpolated average precision (11-point)    | `Evaluation/imap.py`            |
| **P@K**         | Precision at K top results                   | `Evaluation/precision_a_k.py`   |
| **R-Precision** | Precision when retrieving R relevant docs    | `Evaluation/r_precision.py`     |
| **RR**          | Reciprocal of rank of first relevant doc     | `Evaluation/reciprocal_rank.py` |
| **DCG**         | Discounted Cumulative Gain                   | `Evaluation/dcg.py`             |
| **NDCG**        | Normalized DCG (0-1 scale)                   | `Evaluation/ndcg.py`            |
| **Gain**        | Gain visualization                           | `Evaluation/gain.py`            |

## Usage

### Basic Example: Query a Single Model

```python
from preprocessing.indexer import Indexer
from preprocessing.parser import parser_medline
from retrieval_models.run_model import run_model
import config

# Load index
indexer = Indexer()
indexer.load(str(config.INVERTED_INDEX_PATH))

# Parse query
raw_query = "cancer treatment"
query_vec = indexer.preprocess(raw_query)

# Retrieve documents
rankings = run_model("bm25", indexer, query_vec, rel_docs=[])

# Print top-10 results
for rank, (doc_id, score) in enumerate(rankings[:10], 1):
    print(f"{rank:2d}. Doc {doc_id:5d}: {score:.4f}")
```

### Evaluate on Full Query Set

```python
from preprocessing.parser import parse_relevance
from Evaluation.ndcg import ndcg

# Load relevance judgments
all_relevance = parse_relevance(config.MEDLINE_QRELS)

# For each query
for qid in query_ids:
    rel_docs = all_relevance.get(qid, [])

    # Get rankings from model
    rankings = run_model("bm25", indexer, query_vec, rel_docs)

    # Compute NDCG
    score = ndcg(rankings, rel_docs, k=10)
    print(f"Query {qid}: NDCG@10 = {score:.4f}")
```

### Run Tests

```bash
python test.py
```

## Interactive UI

The Streamlit dashboard (`UI/app.py`) provides:

- **Scalar Metrics**: Compare models side-by-side (MAP, NDCG, etc.)
- **PR Curves**: Precision-Recall curves for all models
- **Gain Charts**: Cumulative gain analysis
- **Model Selection**: Filter by single query or aggregate across set

**Launch:**

```bash
streamlit run UI/app.py
```

## Configuration

All hyperparameters and paths are centralized in [config.py](config.py):

### Paths

```python
MEDLINE_DOCS    = Path("data/MED.ALL")      # Document collection
MEDLINE_QUERIES = Path("data/MED.QRY")      # Queries
MEDLINE_QRELS   = Path("data/MED.REL")      # Relevance judgments
INDEX_CACHE_DIR = Path("results/index_cache")
```

### Model Hyperparameters

```python
BM25_K1 = 1.5                    # BM25 saturation parameter
BM25_B = 0.75                    # BM25 length normalization
LM_SMOOTHING_LAMBDA = 0.5        # Jelinek-Mercer λ
LM_DIRICHLET_MU = 2000           # Dirichlet μ
LSI_K = 300                      # LSI dimensions
```

### Evaluation

```python
K = [5,10]                       # Cutoff for P@K, top-K retrieval
DCG_CUTOFF = 20                  # DCG@K
NDCG_CUTOFF = 20                 # NDCG@K
```

Edit these values to adjust model behavior or evaluation settings.

## Author

Kerrouchi Narimane

## References

- Course material from Dr. BIDA, Professor at USTHB (Université des Sciences et de la Technologie Houari Boumediene)
- MEDLINE Dataset: https://www.nlm.nih.gov/medline/

---

**Questions or Issues?** Contact: [narimaneker19@gmail.com](mailto:narimaneker19@gmail.com)
