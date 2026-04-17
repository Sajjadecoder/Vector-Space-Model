import os
import re
import spacy

nlp = spacy.load("en_core_web_sm")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEXES_DIR = os.path.join(BASE_DIR, "indexes")

def load_stopwords():
    path = os.path.join(DATA_DIR, "stopwords.txt")

    stopwords = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stopwords.add(line.strip().lower())
    return stopwords

def case_folding(text):
    return text.lower()

def tokenize(text):
    return re.findall(r"\b[a-z0-9]+\b", text)

def remove_stopwords(tokens, stopwords):
    filtered_tokens = []

    for t in tokens:
        if t not in stopwords:
            filtered_tokens.append(t)

    return filtered_tokens


def lemmatize(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc if token.lemma_.strip()]

def preprocess_pipeline(text, stopwords):
    text = case_folding(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens, stopwords)
    lemmas = lemmatize(tokens)
    return lemmas