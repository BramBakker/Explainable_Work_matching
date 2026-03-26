import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import lemon
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re
def expand_strict_fields(df, column_name, field_list):
    def unflatten_strict(text):
        if pd.isna(text) or text == "":
            return pd.Series({}, dtype='object')

        # 1. Find the start position of every known field in the text
        hits = []
        for field in field_list:
            start = text.find(field)
            if start != -1:
                hits.append((start, field))
        
        # 2. Sort hits by their position in the string
        hits.sort()
        
        # 3. Slice the string based on these positions
        result = {}
        for i in range(len(hits)):
            current_pos, current_field = hits[i]
            
            # The value starts after the field name and ends at the next field's start
            val_start = current_pos + len(current_field)
            val_end = hits[i+1][0] if i+1 < len(hits) else len(text)
            
            # Clean up: remove leading/trailing spaces and the trailing period/comma
            value = text[val_start:val_end].strip().rstrip('.').rstrip(',')
            
            # Remove the colon from the key name for the column header
            clean_key = current_field.rstrip(':').strip()
            result[clean_key] = value
            
        return pd.Series(result)

    return df[column_name].apply(unflatten_strict)

# 2. Prediction Function
def predict_proba(records_a, records_b, record_id_pairs):
    def flatten(row):
        return ". ".join([f"{k}: {v}" for k, v in row.items() if pd.notna(v)])
    # Convert incoming records to strings
    list_a = records_a.apply(flatten, axis=1).tolist()
    list_b = records_b.apply(flatten, axis=1).tolist()
    emb_a = model.encode(list_a, convert_to_tensor=True)
    emb_b = model.encode(list_b, convert_to_tensor=True)
    sim_matrix = cosine_similarity(emb_a, emb_b)
    similarities = []
    for _, pair in record_id_pairs.iterrows():
        a_idx = pair["a.rid"]
        b_idx = pair["b.rid"]
        similarities.append(sim_matrix[a_idx, b_idx])

    similarities = np.array(similarities)
    return similarities

known_fields = ["taal:", "titel:","ondertitel:", "hoofdauteur:", "2e auteur:","editie:","omschrijving:", "vertaling van:"]



model = SentenceTransformer('new_forty_matcher')
with open("realpairs.txt") as texts:
    line_status=0
    pairs=[]
    for line in texts:
        line_status+=1
        if line_status==1:
            pair=[line.strip("\n")]
        if line_status==2:
            pair.append(line.strip("\n"))
        if line_status==3:
            line_status=0
            pairs.append(pair)

number=0
for p in pairs:
    number+=1
    name="explanations/LEMON_pair_"+str(number)
    records_a_orig = pd.DataFrame([
    {
    "raw":p[0]
    }
    ])

    records_b_orig = pd.DataFrame([

    {"raw":p[1]
    }

    ])
    records_a = expand_strict_fields(records_a_orig, 'raw',known_fields)
    records_b = expand_strict_fields(records_b_orig, 'raw',known_fields)

    record_id_pairs = pd.DataFrame({
        "a.rid": [0],
        "b.rid": [0]
    })

    exp = lemon.explain(records_a, records_b, record_id_pairs,predict_proba,granularity="attributes",num_samples=100)
    html_content = exp.as_html()
    with open(name, "w") as f:
        f.write(html_content)
