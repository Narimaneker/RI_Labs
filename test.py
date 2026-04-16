from preprocessing.parser import parser_medline
from preprocessing.indexer import Indexer
from preprocessing.stemmer import stem
from preprocessing.tokenizer import tokenize

docs = parser_medline("data/MED.ALL")
idx = Indexer()
# print(stem(tokenize("the slopes of regression lines")))
idx.build(docs)
print(idx.inverted_index["regress"][1]['tfidf'])
