from preprocessing.indexer import Indexer
from preprocessing.parser import parser_medline
import os


def main():
    indexer = Indexer()

    if os.path.exists("results/index_cache/index.pkl"):
        indexer.load("results/index_cache/index.pkl")
    else: 
        docs = parser_medline("data/MED.ALL")
        indexer.build(docs)
        indexer.save("results/index_cache/index.pkl")
       
if __name__ == "__main__":
    main()   
    
