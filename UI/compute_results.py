"""
Run this ONCE from the project root:
    python UI/compute_results.py

Saves results to UI/results.json
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.parser import parser_medline, parse_relevance
from preprocessing.indexer import Indexer
from retrieval_models.run_model import run_model

# ── config ────────────────────────────────────────────────────────────────────
DOCS_PATH    = "../data/MED.ALL"
QUERIES_PATH = "../data/MED.QRY"
REL_PATH     = "../data/MED.REL"
OUTPUT_PATH  = os.path.join(os.path.dirname(__file__), "results.json")

MODEL_IDS = [
    "vsm", "bm25", "bir", "bir_rel",
    "ext_bir", "ext_bir_rel", "lsi",
    "lm_mle", "lm_laplace", "lm_jm", "lm_dirichlet",
]

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading documents and building index...")
docs      = parser_medline(DOCS_PATH)
queries   = parser_medline(QUERIES_PATH)
relevance = parse_relevance(REL_PATH)

indexer = Indexer()
indexer.build(docs)
print(f"  Index built — {len(docs)} docs, {len(queries)} queries\n")

# ── run models ────────────────────────────────────────────────────────────────
results = {}

for model_id in MODEL_IDS:
    print(f"[{model_id}]")
    results[model_id] = {}

    for q_id, query_text in queries.items():
        query_vec = indexer.preprocess(query_text)
        rel_docs  = relevance.get(q_id, [])

        t0 = time.time()
        try:
            ranked = run_model(model_id, indexer, query_vec, rel_docs)
        except Exception as e:
            print(f"  ✗ query {q_id}: {e}")
            ranked = []
        elapsed = time.time() - t0

        results[model_id][str(q_id)] = {        # JSON keys must be strings
            "ranked":   ranked,                  # [(doc_id, rsv), ...]
            "relevant": rel_docs,
        }
        print(f"  query {q_id:>3} — {len(ranked)} docs ranked ({elapsed:.1f}s)")

    print()

# ── save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f)

print(f"✓ Saved to {OUTPUT_PATH}")