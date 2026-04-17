import os
import json
from preprocess import (
    load_stopwords,
    preprocess_pipeline,
    DATA_DIR,
    INDEXES_DIR
)


def create_vocab_list():
    folder_path = os.path.join(DATA_DIR, "speeches")
    output_file = os.path.join(INDEXES_DIR, "vocabulary.json")

    stopwords = load_stopwords()
    vocab = set()

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        lemmas = preprocess_pipeline(text, stopwords)

        vocab.update(lemmas)

    vocab_list = sorted(list(vocab))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, indent=4)

    return vocab_list


vocab = create_vocab_list()
print("Vocabulary size:", len(vocab))
print("Sample:", vocab[:20])