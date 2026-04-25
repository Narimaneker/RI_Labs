from preprocessing.tokenizer import tokenize
from preprocessing.stopwords import remove_stopwords
from preprocessing.stemmer import stem
import numpy as np
import pickle

# helpers

def tf(tf_raw: int, doc_length: int) -> float:
    return tf_raw / doc_length

def idf(doc_count: int, doc_freq: int) -> float:
    return np.log10(doc_count / doc_freq + 1)

def tfidf(tf_raw: int, doc_length: int, doc_count: int, doc_freq: int) -> float:
    return tf(tf_raw, doc_length) * idf(doc_count, doc_freq)

class Indexer:
    def __init__(self):
        self.inverted_index = {}   # {term: {doc_id: tf}}
        self.doc_lengths = {}      # {doc_id: number of tokens}
        self.avgdl = 0.0           # average document length (for BM25, Dirichlet)
        self.vocab = set()         # all unique terms
        self.doc_count = 0         # total number of documents
        self.vectorizer = None     # fitted TfidfVectorizer (to transform queries)
        self.doc_ids = []          # ordered list of doc_ids (maps matrix rows → doc_id)
        self.tfidf_matrix = None   # np.ndarray shape: (doc_count, vocab_size)
        self.term_index = {}       # {term: column_index}
        self.terms = []            # ordered list of terms
        self.idf = {}              # {term: idf_value}    

    def preprocess(self, text: str) -> list[str]:
        return stem(remove_stopwords(tokenize(text)))
    
    def build(self, docs: dict[int, str])-> None:
        self.doc_ids = list(docs.keys())
        self.doc_count = len(docs)
        preprocessed = {}

        for doc_id, text in docs.items():
            #tokenize -> remove stopwords -> stemming
            tokens = self.preprocess(text)
            self.doc_lengths[doc_id] = len(tokens)
            preprocessed[doc_id] = tokens

            for term in tokens:
                self.vocab.add(term)
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                if doc_id not in self.inverted_index[term]:
                    self.inverted_index[term][doc_id] = {"freq": 0, "tfidf": 0.0}
                self.inverted_index[term][doc_id]["freq"] += 1
        self.avgdl = sum(self.doc_lengths.values()) / self.doc_count
        
        self.terms = list(self.vocab)
        self.term_index = {term: i for i, term in enumerate(self.terms)}
        self.tfidf_matrix = np.zeros((self.doc_count + 1, len(self.terms)))

        for term, doc_tf in self.inverted_index.items():
            doc_freq = len(doc_tf)
            self.idf[term] = idf(self.doc_count, doc_freq)
            col = self.term_index[term]

            for doc_id, values in doc_tf.items():
                row = self.doc_ids.index(doc_id)
                score = tfidf(values["freq"], self.doc_lengths[doc_id], self.doc_count, doc_freq) 
                self.inverted_index[term][doc_id]["tfidf"] = score
                self.tfidf_matrix[row][col] = score       


    def vectorize_query(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.terms))

        for term in set(tokens):  # set → avoid duplicates
            if term in self.term_index:
                vec[self.term_index[term]] = 1.0

        return vec


    def save(self, path:str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)


    def load(self, path:str) -> None:
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))