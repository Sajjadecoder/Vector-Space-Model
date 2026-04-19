import os
import json
from preprocess import INDEXES_DIR


def compute_tfidf():
    tf_file = os.path.join(INDEXES_DIR, "tf.json")
    idf_file = os.path.join(INDEXES_DIR, "idf.json")
    output_file = os.path.join(INDEXES_DIR, "tfidf.json")

    with open(tf_file, "r", encoding="utf-8") as f:
        tf = json.load(f)

    with open(idf_file, "r", encoding="utf-8") as f:
        idf = json.load(f)

    tfidf = {}

    for doc_id in tf:
        temp = {}

        for term in tf[doc_id]:
            if term in idf:
                temp[term] = tf[doc_id][term] * idf[term]

        tfidf[doc_id] = temp

    tfidf = dict(sorted(tfidf.items(), key=lambda x: int(x[0])))

    # save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tfidf, f, indent=4)

    return tfidf


compute_tfidf()
