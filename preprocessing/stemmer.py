from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')

_stemmer = PorterStemmer()

def stem(tokens: list[str]) -> list[str]:
    return [_stemmer.stem(t) for t in tokens]