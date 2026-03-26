import torch
import numpy as np
from helper_plots import plot_bilrp_sentences
import re
from sklearn.preprocessing import normalize

def extract_content_from_string(raw_string, fieldlist):
    """
    Extracts content using only raw strings and the fieldlist.
    No tokens, no indices, no special characters.
    """
    # 1. Clean up the raw string (remove <s>, </s>, and the ▁ marker)
    clean_text = raw_string.replace('<s>', '').replace('</s>', '').replace('▁', '').strip()
    
    # 2. Prepare the fieldlist (ensure they have colons and are escaped for Regex)
    # We sort by length descending so "vertaling van:" is matched before "vertaling:"
    sorted_fields = sorted([f.strip() for f in fieldlist], key=len, reverse=True)
    
    # 3. Build a Regex pattern that matches any of the fields
    # Pattern looks like: (field1:|field2:|field3:)
    field_pattern = '|'.join([re.escape(f) for f in sorted_fields])
    
    # 4. Split the string by the field labels, but keep the labels in the result
    # We use a lookahead/split combo to get [Label, Content, Label, Content...]
    parts = re.split(f'({field_pattern})', clean_text, flags=re.IGNORECASE)
    
    content_dict = {}
    current_label = None
    
    for part in parts:
        p = part.strip()
        if not p:
            continue
            
        # Check if this part is one of our labels
        if any(p.lower() == f.lower() for f in sorted_fields):
            current_label = p.lower()
            if not current_label.endswith(':'):
                current_label += ':'
        else:
            # If we have a label active, this part is the content
            if current_label:
                # If content already exists (e.g. multiple snippets), append it
                if current_label in content_dict:
                    content_dict[current_label] += " " + p
                else:
                    content_dict[current_label] = p
                    
    return content_dict

def get_field_map(tokens, fieldlist):
    print(tokens)
    field_map = {}      
    current_field = None
    ignored_markers = ['<s>', '</s>']
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ignored_markers:
            i += 1
            continue
        fullword = token.replace('▁', ' ')
        
        j = i + 1
        while j < len(tokens) and '▁' not in tokens[j]:
            fullword += tokens[j]
            j += 1
        found_field = None
        clean_word = fullword.strip().lower()
        if clean_word in fieldlist:
            found_field = clean_word
        elif clean_word == "vertaling" or clean_word == "2e":
            k = j 
            if k < len(tokens):
                next_word = tokens[k].replace('▁', ' ')
                k += 1
                while k < len(tokens) and '▁' not in tokens[k]:
                    next_word += tokens[k]
                    k += 1
                if next_word.strip().lower() == "van:":
                    found_field = "vertaling van:"
                if next_word.strip().lower() == "auteur:":
                    found_field = "2e auteur:"
        if found_field:
            current_field = found_field
    
        if current_field not in field_map:
            field_map[current_field] = []
        
        field_map[current_field].append(i)
        i += 1            
    return field_map



def aggregate_matrix_old(matrix, tokens_a, tokens_b, fieldlist, dictext_a, dictext_b):
    
    map_a = get_field_map(tokens_a, fieldlist)
    map_b = get_field_map(tokens_b, fieldlist)

    # 2. Create the reduced matrix
    field_names_a = list(map_a.keys())
    field_names_b = list(map_b.keys())
    filled_fields_a = []
    filled_fields_b = []

    # Using the passed dictext variables instead of looking for globals
    for f in field_names_a:
        fieldcontent = dictext_a.get(f, "") # .get prevents crashes if field is missing
        fullfield = f + " " + fieldcontent
        filled_fields_a.append(fullfield.replace("@", ""))
    for f in field_names_b:
        fieldcontent = dictext_b.get(f, "")
        fullfield = f + " " + fieldcontent
        filled_fields_b.append(fullfield.replace("@", ""))

    reduced_matrix = np.zeros((len(field_names_a), len(field_names_b)))

    for i, f_a in enumerate(field_names_a):
        for j, f_b in enumerate(field_names_b):
            indices_a = map_a[f_a]
            indices_b = map_b[f_b]
            block = matrix[np.ix_(indices_a, indices_b)]
            reduced_matrix[i, j] = block.mean()

    rsen_a = np.tile(reduced_matrix.sum(axis=1).reshape(1, 1, -1), (1, 2, 1))
    rsen_b = np.tile(reduced_matrix.sum(axis=0).reshape(1, 1, -1), (1, 2, 1))
    
    outs = [{'Rsen': rsen_a}, {'Rsen': rsen_b}]
    return reduced_matrix, filled_fields_a, filled_fields_b, outs


def aggregate_matrix(matrix, tokens_a, tokens_b, fieldlist, dictext_a, dictext_b):
    # 1. Map tokens to fields (This creates the numerical indices)
    map_a = get_field_map(tokens_a, fieldlist)
    map_b = get_field_map(tokens_b, fieldlist)

    field_names_a = list(map_a.keys())
    field_names_b = list(map_b.keys())
    filled_fields_a = []
    filled_fields_b = []

    # Format the labels for the heatmap axes
    for f in field_names_a:
        fieldcontent = dictext_a.get(f, "")
        fullfield = f + " " + fieldcontent
        filled_fields_a.append(fullfield.replace("@", ""))
    for f in field_names_b:
        fieldcontent = dictext_b.get(f, "")
        fullfield = f + " " + fieldcontent
        filled_fields_b.append(fullfield.replace("@", ""))

    reduced_matrix = np.zeros((len(field_names_a), len(field_names_b)))

    for i, f_a in enumerate(field_names_a):
        for j, f_b in enumerate(field_names_b):
            # FIX 1: Flatten indices to 1D to avoid "Cross index" error
            # This handles cases where get_field_map returns nested lists
            idx_a = np.hstack(map_a[f_a]).astype(int) if len(map_a[f_a]) > 0 else np.array([])
            idx_b = np.hstack(map_b[f_b]).astype(int) if len(map_b[f_b]) > 0 else np.array([])
            
            # FIX 2: Only calculate mean if there is data in the block
            if idx_a.size > 0 and idx_b.size > 0:
                block = matrix[np.ix_(idx_a, idx_b)]
                reduced_matrix[i, j] = block.mean()
            else:
                reduced_matrix[i, j] = 0

    # FIX 3: Tile by 768 instead of 2. 
    # This ensures that when the plotter does .squeeze()[0, :], it always succeeds.
    hidden_size = 768
    rsen_a = np.tile(reduced_matrix.sum(axis=1).reshape(1, 1, -1), (1, hidden_size, 1))
    rsen_b = np.tile(reduced_matrix.sum(axis=0).reshape(1, 1, -1), (1, hidden_size, 1))
    
    outs = [{'Rsen': rsen_a}, {'Rsen': rsen_b}]
    return reduced_matrix, filled_fields_a, filled_fields_b, outs


def get_explanation(text_a, text_b,model,name):
    # Load model once (or pass it in)
    fieldlist = ['taal:', 'titel:', 'hoofdauteur:', 'ondertitel:', 'omschrijving:', 'editie:']
    fieldlist_str = ['taal:', 'titel:', 'hoofdauteur:', 'ondertitel:', '2e auteur', 'vertaling van:', 'omschrijving:', 'editie:', 'auteur/primair']
    
    dictext_a = extract_content_from_string(text_a, fieldlist_str)
    dictext_b = extract_content_from_string(text_b, fieldlist_str)
    # 3. Tokenize and get Embeddings
    tokens_a_res = model.tokenizer(text_a, return_tensors="pt")
    tokens_b_res = model.tokenizer(text_b, return_tensors="pt")

    input_ids_a = tokens_a_res['input_ids']

    input_ids_b = tokens_b_res['input_ids']
    mask_a = tokens_a_res['attention_mask']
    mask_b = tokens_b_res['attention_mask']
    out_a = model.forward_explain(input_ids_a, mask_a)
    out_b = model.forward_explain(input_ids_b, mask_b)
    print(out_b['Rsen'])
    interaction_matrix = out_a['Rsen'][0].squeeze().T @ out_b['Rsen'][0].squeeze()
    interaction_matrix /= np.abs(interaction_matrix).max()
    # 6. Prepare tokens for aggregation
    tokens_a = model.tokenizer.convert_ids_to_tokens(input_ids_a[0])
    tokens_b = model.tokenizer.convert_ids_to_tokens(input_ids_b[0])
    # 7. Aggregate and Plot
    # Added the extra arguments here!
    res_matrix, fields_a, fields_b, outs = aggregate_matrix(
        interaction_matrix, tokens_a, tokens_b, fieldlist, dictext_a, dictext_b
    )

    normed_matrix=normalize(res_matrix, axis=1, norm='l1')
    
    plot_bilrp_sentences(
        decoded=[fields_a, fields_b],
        outs=outs,
        relevance_scores=normed_matrix,
        title="Field-Level Interaction",
        yticks_fontsize=13,
        name=name
    )


def get_baseline_explanation(text_a, text_b,model,name):
    # Load model once (or pass it in)
    fieldlist_str = ['taal:', 'titel:', 'hoofdauteur:', 'ondertitel:', '2e auteur', 'vertaling van:', 'omschrijving:', 'editie:', 'auteur/primair']
    
    dictext_a = extract_content_from_string(text_a, fieldlist_str)
    dictext_b = extract_content_from_string(text_b, fieldlist_str)
    embs_a=[]
    fields_a=[]
    fields_b=[]
    for key_a in dictext_a.keys():
        comb_string_a=key_a+dictext_a[key_a]
        fields_a.append(comb_string_a)
        emb_a=model.encode(comb_string_a)
        embs_a.append(emb_a)

    embs_b=[]

    for key_b in dictext_b.keys():
        comb_string_b=key_b+dictext_b[key_b]
        fields_b.append(comb_string_b)
        emb_b=model.encode(comb_string_b)
        embs_b.append(emb_b)

    interaction_matrix = np.array(embs_a) @ np.array(embs_b).T

    rsen_a = np.tile(interaction_matrix.sum(axis=1).reshape(1, 1, -1), (1, 2, 1))
    rsen_b = np.tile(interaction_matrix.sum(axis=0).reshape(1, 1, -1), (1, 2, 1))
    
    outs = [{'Rsen': rsen_a}, {'Rsen': rsen_b}]

    plot_bilrp_sentences(
        decoded=[fields_a, fields_b],
        outs=outs,
        relevance_scores=interaction_matrix,
        title="Field-Level Interaction",
        yticks_fontsize=13,
        name=name
    )




