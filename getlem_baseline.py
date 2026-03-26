import pandas as pd
import numpy as np
from difflib import SequenceMatcher # Built-in, or use 'python-Levenshtein' for speed
from lemon._matching_attribution_explanation import MatchingAttributionExplanation, Attribution


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


known_fields = ["taal:", "titel:","ondertitel:", "hoofdauteur:", "2e auteur:","editie:","omschrijving:", "vertaling van:"]



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


def get_similarity(str1, str2):
    if pd.isna(str1) or pd.isna(str2): return 0.0
    return SequenceMatcher(None, str(str1), str(str2)).ratio()

def create_max_sim_explanation(records_a, records_b, pair_idx=0):
    row_a = records_a.iloc[pair_idx]
    row_b = records_b.iloc[pair_idx]
    all_cols = sorted(list(set(records_a.columns) | set(records_b.columns)))
    
    string_rep = {}
    for col in all_cols:
        # Pre-fill strings to avoid KeyErrors
        val_a = row_a[col] if col in records_a.columns else np.nan
        val_b = row_b[col] if col in records_b.columns else np.nan
        string_rep[("a", col, "attr")] = col
        string_rep[("a", col, "val")] = None if pd.isna(val_a) else str(val_a)
        string_rep[("b", col, "attr")] = col
        string_rep[("b", col, "val")] = None if pd.isna(val_b) else str(val_b)

    attributions = []
    # Keep track of which B-fields have already been assigned an attribution
    assigned_b = set()

    # 1. Match A-fields to B-fields
    for col_a in records_a.columns:
        val_a = row_a[col_a]
        if pd.isna(val_a): continue
            
        best_sim = 0.0
        best_col_b = None
        for col_b in records_b.columns:
            # ONLY consider B-fields not already assigned to avoid the AssertionError
            if col_b in assigned_b: continue
            
            sim = get_similarity(val_a, row_b[col_b])
            if sim > best_sim:
                best_sim = sim
                best_col_b = col_b
        
        if best_col_b and best_sim > 0.1: # Threshold to avoid noisy weak matches
            attributions.append(Attribution(
                weight=best_sim,
                positions=[("a", col_a, "val", None), ("b", best_col_b, "val", None)],
                potential=0
            ))
            assigned_b.add(best_col_b)
        else:
            # If no match in B, A gets its own zero-weight attribution (white highlight)
            attributions.append(Attribution(weight=0.0, positions=[("a", col_a, "val", None)], potential=0))

    # 2. Catch remaining B-fields that weren't matched
    for col_b in records_b.columns:
        if col_b not in assigned_b and not pd.isna(row_b[col_b]):
            attributions.append(Attribution(weight=0.0, positions=[("b", col_b, "val", None)], potential=0))

    # 3. Create DataFrame
    df_a = records_a.iloc[[pair_idx]].reindex(columns=all_cols)
    df_b = records_b.iloc[[pair_idx]].reindex(columns=all_cols)
    record_pair = pd.concat([df_a, df_b], axis=1, keys=['a', 'b'])

    return MatchingAttributionExplanation(
        record_pair=record_pair,
        string_representation=string_rep,
        attributions=attributions,
        prediction_score=np.mean([a.weight for a in attributions if a.weight > 0]) if attributions else 0,
        dual=True
    )
number=0
for p in pairs:
    number+=1
    print('hey')
    name="explanations/LEMON_baseline_"+str(number)+".html"
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

    html_content=create_max_sim_explanation(records_a,records_b).as_html()
    with open(name, "w") as f:
        f.write(html_content)
