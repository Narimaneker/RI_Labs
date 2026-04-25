from preprocessing.indexer import Indexer
from preprocessing.parser import parser_medline, parse_relevance
from retrieval_models.run_model import run_model
import os


import config

def main():
    index_path = config.INDEX_CACHE_DIR / "index.pkl"
    indexer = Indexer()

    if index_path.exists():
        print(f"[+] Loading index from {index_path} ...")
        indexer.load(str(index_path))
    else:
        print("[+] Building index ...")
        docs = parser_medline(str(config.MEDLINE_DOCS))
        indexer.build(docs)
        indexer.save(str(index_path))
        print(f"    Saved to {index_path}")
    
    print(f"    Index ready — {indexer.doc_count} docs\n")
     # 2. Load queries and relevance judgments
    print(f"[+] Loading queries   : {config.MEDLINE_QUERIES}")
    all_queries   = parser_medline(config.MEDLINE_QUERIES)
    print(f"[+] Loading relevance : {config.MEDLINE_QRELS}")
    all_relevance = parse_relevance(config.MEDLINE_QRELS)

    # query_ids = config.EVAL_QUERY_IDS
    query_ids = [1]
    print(f"    Evaluating {len(query_ids)} quer{'y' if len(query_ids) == 1 else 'ies'}\n")
 
    # 3. Run each query through all 11 models
    sep = "=" * 72
    for qid in query_ids:
        raw_query = all_queries.get(qid)
        if raw_query is None:
            print(f"[!] Query {qid} not found in MED.QRY, skipping.\n")
            continue
 
        query_vec = indexer.preprocess(raw_query)
        rel_docs  = all_relevance.get(qid, [])
        rel_set   = set(rel_docs)   # O(1) lookup for ✓ marking
 
        print(sep)
        print(f"  QUERY {qid:>3} : {raw_query[:80].strip()}{'...' if len(raw_query) > 80 else ''}")
        print(f"  Tokens    : {query_vec}")
        print(f"  Relevant  : {len(rel_docs)} docs in MED.REL")
        print(sep)
 
        for model_id in config.MODEL_IDS:
            label = config.MODEL_LABELS[model_id]
            try:
                results = run_model(model_id, indexer, query_vec, rel_docs)
                top     = results[:config.TOP_K]
 
                print(f"\n  [{label}]")
                print(f"  {'Doc ID':<10} {'Score':<14} Relevant?")
                print(f"  {'-'*38}")
                for doc_id, score in top:
                    mark = "✓" if doc_id in rel_set else ""
                    print(f"  {str(doc_id):<10} {score:<14.6e} {mark}")
 
            except Exception as e:
                print(f"\n  [{label}]")
                print(f"  ERROR: {e}")
 
        print()
 
    print(sep)
    print("  Done.")
    print(sep)
   
       
if __name__ == "__main__":
    main()   
    
