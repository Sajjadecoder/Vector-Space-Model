import json
import os
from preprocess import INDEXES_DIR

def compute_tf():
    input_file = os.path.join(INDEXES_DIR, "positional_index.json")
    output_file = os.path.join(INDEXES_DIR, "tf.json")

    with open(input_file, "r", encoding="utf-8") as f:
        positional_index = json.load(f)

    tf = {}

    for term in positional_index:
        for doc_id in positional_index[term]:
            if doc_id not in tf:
                tf[doc_id] = {}

            tf[doc_id][term] = len(positional_index[term][doc_id])

    # save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tf, f, indent=4)

compute_tf()