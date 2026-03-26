import re

import re

included_fields =["taal", "titel","ondertitel", "hoofdauteur", "2e auteur","editie","omschrijving", "vertaling van"]


with open("final_output.txt", "r") as infile, open("real_train.txt", "w") as outfile:
    for line in infile:
        tab_items = line.strip().split("\t")
        
        *items, label = tab_items  # last element = label
        
        new_items = []
        
        for item in items:
            parts = [p.strip() for p in item.split(".") if p.strip()]
            
            filtered_parts = []
            for part in parts:
                if ": " in part:
                    key, value = part.split(": ", 1)
                    if key.strip() in included_fields:
                        filtered_parts.append(f"{key.strip()}: {value.strip()}")
            
            if filtered_parts:
                new_items.append(". ".join(filtered_parts))
        
        # Rebuild line: items + label
        if new_items:
            outfile.write("\t".join(new_items + [label]) + "\n")