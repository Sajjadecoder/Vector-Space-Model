import os
import json
from preprocess import INDEXES_DIR

def compute_df():
    inverted_index_directory = os.path.join(INDEXES_DIR,"inverted_index.json")
    output_file = os.path.join(INDEXES_DIR,"df.json")
    with open(inverted_index_directory, "r", encoding="utf-8") as f:
        inverted_index = json.load(f)
    
    df = {}
    for term in inverted_index:
        term_df = len(inverted_index[term])
        df[term] = term_df
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(df, f, indent=4)
        
        
compute_df()
