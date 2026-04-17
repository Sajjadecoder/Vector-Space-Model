import math
import json
import os
from preprocess import INDEXES_DIR

def compute_idf():
    df_file = os.path.join(INDEXES_DIR, "df.json")
    output_file = os.path.join(INDEXES_DIR, "idf.json")
    total_docs = 56
    with open(df_file, "r", encoding="utf-8") as f:
        df = json.load(f)
    
    idf = {}
    for term in df:
        df_val = df[term]
        if df_val == 0:
            idf[term] = 0
            
        else:
            idf[term] = math.log(total_docs / df_val)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(idf, f, indent=4)

compute_idf()