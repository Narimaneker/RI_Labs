import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

StopWords=stopwords.words('english')

def remove_stopwords(tokens: list[str])-> list[str]:
    return [t for t in tokens if t not in StopWords]