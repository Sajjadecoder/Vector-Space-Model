import os
import json
import re
from preprocess import (
    load_stopwords,
    preprocess_pipeline,
    DATA_DIR,
    INDEXES_DIR
)


def positional_index_generation():
    folder_path = os.path.join(DATA_DIR, "speeches")
    output_file = os.path.join(INDEXES_DIR, "positional_index.json")

    stopwords = load_stopwords()
    positional_index = {}

    doc_id = 0

    files = sorted(os.listdir(folder_path), key=lambda x: int(re.search(r'\d+', x).group()))

    for file in files:
        file_path = os.path.join(folder_path, file)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = preprocess_pipeline(text, stopwords)

        for position, term in enumerate(tokens):

            if term not in positional_index:
                positional_index[term] = {}

            if doc_id not in positional_index[term]:
                positional_index[term][doc_id] = []

            positional_index[term][doc_id].append(position)

        doc_id += 1 

    for term in positional_index:
        for d in positional_index[term]:
            positional_index[term][d] = sorted(positional_index[term][d])

        positional_index[term] = dict(sorted(positional_index[term].items()))

    positional_index = dict(sorted(positional_index.items()))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(positional_index, f, indent=4)

    return positional_index


index = positional_index_generation()
print("Positional index size:", len(index))