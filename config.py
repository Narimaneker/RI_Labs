"""
config.py – Central configuration for IR Lab 5
All paths, hyperparameters, and constants live here.
Import this module everywhere instead of hardcoding values.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

ROOT_DIR        = Path(__file__).resolve().parent
DATA_DIR        = ROOT_DIR / "data"
RESULTS_DIR     = ROOT_DIR / "results"
RANKINGS_DIR    = RESULTS_DIR / "rankings"
PLOTS_DIR       = RESULTS_DIR / "plots"
NOTEBOOKS_DIR   = ROOT_DIR / "notebooks"

# Raw MEDLINE collection files
MEDLINE_DOCS    = DATA_DIR / "MED.ALL"
MEDLINE_QUERIES = DATA_DIR / "MED.QRY"
MEDLINE_QRELS   = DATA_DIR / "MED.REL"

# Serialized index cache (avoids recomputing on every run)
INDEX_CACHE_DIR = RESULTS_DIR / "index_cache"
INVERTED_INDEX_PATH   = INDEX_CACHE_DIR / "inverted_index.pkl"
DOC_TERM_MATRIX_PATH  = INDEX_CACHE_DIR / "doc_term_matrix.pkl"
TF_MATRIX_PATH        = INDEX_CACHE_DIR / "tf_matrix.pkl"
TFIDF_MATRIX_PATH     = INDEX_CACHE_DIR / "tfidf_matrix.pkl"
VOCAB_PATH            = INDEX_CACHE_DIR / "vocab.pkl"
DOC_IDS_PATH          = INDEX_CACHE_DIR / "doc_ids.pkl"

# Create output directories if they don't exist yet
for _dir in [RESULTS_DIR, RANKINGS_DIR, PLOTS_DIR, INDEX_CACHE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

# Tokenization regex: keeps only alphabetic tokens of length >= 2
TOKENIZER_PATTERN = (
    r'(?:[A-Za-z]\.)+'
    r'|[A-Za-z]+[\-@]\d+(?:\.\d+)?'
    r'|\d+(?:[\.\,\-]\d+)*%?'
    r'|[A-Za-z]+'
)

# Path to custom stopword list (one word per line).
# Set to None to use the built-in NLTK English stopwords.
STOPWORDS_FILE = None  # e.g. DATA_DIR / "stopwords.txt"

# Stemmer to use: "porter" (default) or "snowball"
STEMMER = "porter"

# Minimum document frequency for a term to be kept in the vocabulary.
# Terms appearing in fewer than MIN_DF documents are discarded.
MIN_DF = 1


# ─────────────────────────────────────────────
# TF–IDF formula (per lecture notes)
# ─────────────────────────────────────────────
# TF  = raw count  (no sublinear scaling unless overridden)
# IDF = log10(N / df)   — N = total number of documents
# Use sublinear_tf=True to apply 1 + log(tf) instead of raw tf

TFIDF_SUBLINEAR_TF = False    # set True to match some lecture variants


# ─────────────────────────────────────────────
# Retrieval model hyperparameters
# ─────────────────────────────────────────────

# --- Vector Space Model ---
VSM_SIMILARITY = "cosine"          # only cosine is required

# --- Latent Semantic Indexing ---
LSI_K = 100                        # number of latent dimensions (SVD components)

# --- BM25 ---
BM25_K1 = 1.2                      # term frequency saturation
BM25_B  = 0.75                     # document length normalisation

# --- Language Models ---
LM_SMOOTHING_LAMBDA  = 0.2         # Jelinek–Mercer: weight on collection model
LM_SMOOTHING_ADD     = 1           # Laplace (Add-k): k value
# Dirichlet µ is computed as the average document length in the collection.
# Set to None to trigger auto-computation at runtime; set an int to override.
LM_DIRICHLET_MU      = None        # e.g. 2000 to hard-code

# Log-domain computation for language models (prevents underflow)
LM_LOG_DOMAIN = True


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

# Queries to include in the evaluation report (1-indexed, matching MED.QRY IDs)
EVAL_QUERY_IDS = list(range(1, 31))          # all 30 queries
REPORT_QUERY_IDS = list(range(1, 11))        # first 10 for the report (I1–I10)

# Gain (%) comparison: queries and metric
GAIN_QUERY_IDS = list(range(1, 11))
GAIN_METRIC    = "ndcg@20"

# P@K cutoffs
PRECISION_AT_K = [5, 10]

# DCG / nDCG cutoff
DCG_CUTOFF  = 20
NDCG_CUTOFF = 20

# Standard recall levels for interpolated P-R curves
INTERPOLATED_RECALL_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                               0.6, 0.7, 0.8, 0.9, 1.0]

# Number of results to retrieve per query (ranking depth)
RETRIEVAL_CUTOFF = 1000


# ─────────────────────────────────────────────
# Model registry
# Used by the evaluator and UI to iterate over all models.
# Each entry: (model_id, display_name, module_path, class_name, kwargs)
#   module_path : importable dotted path matching your folder structure
#   class_name  : the class to instantiate inside that module
# ─────────────────────────────────────────────

MODEL_REGISTRY = [
    # model_id        display_name                              module_path                                          class_name    kwargs
    ("vsm",         "VSM (Cosine)",                            "retrieval_models.vector_space_model.cosinsim",      "VSM",        {}),
    ("lsi",         f"LSI (k={LSI_K})",                       "retrieval_models.Latent_Semantic_Indexing.lsi",     "LSI",        {"k": LSI_K}),
    ("bir",         "BIR (no relevance)",                      "retrieval_models.BIR.bir",                          "BIR",        {}),
    ("bir_rel",     "BIR (with relevance)",                    "retrieval_models.BIR.bir_rel",                      "BIR_Rel",    {}),
    ("ext_bir",     "Extended BIR (no relevance)",             "retrieval_models.BIR.ex_bir",                       "ExtBIR",     {}),
    ("ext_bir_rel", "Extended BIR (with relevance)",           "retrieval_models.BIR.ex_bir_rel",                   "ExtBIR_Rel", {}),
    ("bm25",        f"BM25 (k1={BM25_K1}, b={BM25_B})",       "retrieval_models.BM25.bm25",                        "BM25",       {"k1": BM25_K1, "b": BM25_B}),
    ("lm_mle",      "LM – MLE",                               "retrieval_models.language_model.mle",               "MLE",        {}),
    ("lm_laplace",  "LM – Laplace (Add-1)",                   "retrieval_models.language_model.laplace",           "Laplace",    {"add_k": LM_SMOOTHING_ADD}),
    ("lm_jm",       f"LM – Jelinek–Mercer (λ={LM_SMOOTHING_LAMBDA})",
                                                               "retrieval_models.language_model.jm",                "JM",         {"lam": LM_SMOOTHING_LAMBDA}),
    ("lm_dirichlet","LM – Dirichlet",                         "retrieval_models.language_model.dirichlet",         "Dirichlet",  {"mu": LM_DIRICHLET_MU}),
]

# Convenient lookups
MODEL_IDS    = [m[0] for m in MODEL_REGISTRY]
MODEL_LABELS = {m[0]: m[1] for m in MODEL_REGISTRY}
MODEL_MODULE = {m[0]: m[2] for m in MODEL_REGISTRY}
MODEL_CLASS  = {m[0]: m[3] for m in MODEL_REGISTRY}
MODEL_KWARGS = {m[0]: m[4] for m in MODEL_REGISTRY}


# ─────────────────────────────────────────────
# UI settings
# ─────────────────────────────────────────────

UI_FRAMEWORK        = "streamlit"   # "streamlit" or "pyqt"
UI_RESULTS_PER_PAGE = 20           # ranked documents shown per query in the UI
UI_DEFAULT_QUERY_ID = 1            # query pre-selected on app launch
UI_PLOT_HEIGHT      = 450          # px height for P-R curve plots


# ─────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────

RANDOM_SEED   = 42     # for any stochastic steps (LTR, etc.)
VERBOSE       = True   # print progress to stdout during indexing / evaluation