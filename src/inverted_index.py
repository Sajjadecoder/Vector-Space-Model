import re
import os
import json
from preprocess import (
    load_stopwords,
    preprocess_pipeline,
    DATA_DIR,
    INDEXES_DIR
)


def inverted_index_generation():
    folder_path = os.path.join(DATA_DIR, "speeches")
    output_file = os.path.join(INDEXES_DIR, "inverted_index.json")

    stopwords = load_stopwords()
    inverted_index = {}

    doc_id = 0

    # numeric sorting of files
    files = sorted(os.listdir(folder_path), key=lambda x: int(re.search(r'\d+', x).group()))

    for file in files:
        file_path = os.path.join(folder_path, file)

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = preprocess_pipeline(text, stopwords)
        unique_tokens = set(tokens)

        for term in unique_tokens:
            if term not in inverted_index:
                inverted_index[term] = []

            inverted_index[term].append(doc_id)

        # increment AFTER processing one document
        doc_id += 1

    for term in inverted_index:
        inverted_index[term] = sorted(inverted_index[term])
        
    inverted_index = dict(sorted(inverted_index.items()))
    # save index
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=4)

    return inverted_index


# test run
index = inverted_index_generation()
print("Vocabulary size:", len(index))