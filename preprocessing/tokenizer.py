from nltk.tokenize import RegexpTokenizer
from config import TOKENIZER_PATTERN

_tokenizer = RegexpTokenizer(TOKENIZER_PATTERN)

def tokenize(text: str)->list[str]:
    return _tokenizer.tokenize(text.lower())
