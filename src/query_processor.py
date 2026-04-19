import re
import os
import json
import math
import argparse

from preprocess import load_stopwords, preprocess_pipeline, INDEXES_DIR, DATA_DIR


SPEECHES_DIR = os.path.join(DATA_DIR, "speeches")
ALPHA = 0.005


def load_json(filename):
    path = os.path.join(INDEXES_DIR, filename)

    file = open(path, "r", encoding="utf-8")
    data = json.load(file)
    file.close()

    return data


def load_all_indexes():
    inverted_idx = load_json("inverted_index.json")
    tfidf = load_json("tfidf.json")
    idf = load_json("idf.json")

    files = os.listdir(SPEECHES_DIR)

    # Sort files based on numbers in filename
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        return 0

    files = sorted(files, key=extract_number)

    doc_names = {}

    i = 0
    for fname in files:
        doc_names[str(i)] = fname
        i = i + 1

    return tfidf, idf, inverted_idx, doc_names


def build_query_vector(query_terms, idf):
    query_tf = {}

    for term in query_terms:
        if term in query_tf:
            query_tf[term] = query_tf[term] + 1
        else:
            query_tf[term] = 1

    query_vector = {}

    for term in query_tf:
        tf_val = query_tf[term]

        if term in idf:
            query_vector[term] = tf_val * idf[term]

    return query_vector


def cosine_similarity(query_vector, doc_vec):
    dot_product = 0

    # Calculate dot product
    for term in query_vector:
        if term in doc_vec:
            dot_product = dot_product + (query_vector[term] * doc_vec[term])

    if dot_product == 0.0:
        return 0.0

    sum_q = 0
    for value in query_vector.values():
        sum_q = sum_q + (value * value)
    norm_q = math.sqrt(sum_q)

    sum_d = 0
    for value in doc_vec.values():
        sum_d = sum_d + (value * value)
    norm_d = math.sqrt(sum_d)

    if norm_q == 0.0 or norm_d == 0.0:
        return 0.0

    similarity = dot_product / (norm_q * norm_d)
    return similarity


def get_candidate_docs(query_terms, inverted_idx):
    candidates = set()

    for term in query_terms:
        if term in inverted_idx:
            doc_list = inverted_idx[term]

            for d in doc_list:
                candidates.add(str(d))

    return candidates


# -------------------- RETRIEVE --------------------
def retrieve(raw_query, tfidf, idf, inverted_idx, doc_names, alpha=ALPHA):
    stopwords = load_stopwords()
    query_terms = preprocess_pipeline(raw_query, stopwords)

    if len(query_terms) == 0:
        print("No terms generated from query after preprocessing.")
        return []

    query_vec = build_query_vector(query_terms, idf)

    if len(query_vec) == 0:
        print("Query terms not found in vocabulary.")
        return []

    candidates = get_candidate_docs(query_terms, inverted_idx)

    if len(candidates) == 0:
        print("No documents contain any query term.")
        return []

    scores = []

    for doc_id in candidates:
        if doc_id in tfidf:
            doc_vec = tfidf[doc_id]
        else:
            doc_vec = {}

        score = cosine_similarity(query_vec, doc_vec)

        if score >= alpha:
            filename = doc_names.get(doc_id)

            if filename is None:
                filename = "doc_" + str(doc_id)

            scores.append((doc_id, filename, score))

    # Sort scores
    scores = sorted(scores, key=lambda x: x[2], reverse=True)

    return scores


# -------------------- PRINT RESULTS --------------------
def print_results(query, results):
    print("  Query : " + query)
   
    if len(results) == 0:
        print("  No documents retrieved above the alpha threshold.\n")
        return

    print("  Rank   Score      Doc ID   Filename")
    print("  " + "-" * 56)

    rank = 1
    for item in results:
        doc_id = item[0]
        filename = item[1]
        score = item[2]

        print(f"  {rank:<6} {score:<10.6f} {doc_id:<8} {filename}")
        rank = rank + 1

    print()


def start_execution(tfidf, idf, inverted_idx, doc_names):
    print("   Type 'exit' to stop.")

    while True:
        raw_query = input("  Enter query: ")
        raw_query = raw_query.strip()

        if raw_query.lower() == "exit":
            print("Exiting...")
            break

        if raw_query == "":
            continue

        results = retrieve(raw_query, tfidf, idf, inverted_idx, doc_names)
        print_results(raw_query, results)


def main():
    parser = argparse.ArgumentParser(
        description="VSM Query Processor"
    )

    parser.add_argument("--queries", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()

    print("Loading indexes …", end=" ", flush=True)

    tfidf, idf, inverted_idx, doc_names = load_all_indexes()

    print("done.")
    print("Corpus size : " + str(len(doc_names)) + " documents")
    print("Vocabulary  : " + str(len(idf)) + " terms")
    print("Alpha (threshold): " + str(args.alpha))

   
    start_execution(tfidf, idf, inverted_idx, doc_names)


if __name__ == "__main__":
    main()