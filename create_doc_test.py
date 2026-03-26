import csv
import re
import pickle
from collections import defaultdict


lookup_ppn = {
#    "004Ag": "bindwijze",
    "010@a": "taal",
   # "011@a": "jaar",
   # "011@e": "oorspronkelijk jaar",
  #  "019@0": "land",
    "021Aa": "titel",
    "021Ad": "ondertitel",
    "032@a": "editie",
  #  "033Ap": "plaats",
  #  "033An": "uitgever",
 #   "034Da": "pagina’s",
 #   "034Ia": "formaat",
    "037Aa": "omschrijving",
 #   "004A0": "ISBN",
 #   "004Z8": "trefwoord",
 #   "020Aa": "annotatie",
    "039Dc": "vertaling van",
 #   "137Aa": "lokale annotatie"   
}
test_input = defaultdict(list)

targets = pickle.load(open("id_pairs_unseen.p", "rb"))
test_ids = [t[0] for t in targets]
testing=False
with open("romans_kinderboeken_picaplus_wem.txt", mode ='r', encoding='utf-8')as file:
    next(file) 
    amount=0
    previous_fields = ''
    for lines in file:
        amount+=1
        fields=lines.split("\t")
        if fields[0] != previous_fields:
            if fields[0] in test_ids:
                testing=True
            else:
                testing=False
        if testing==True:    
            if len(fields)==1:
                continue
            sfields=fields[1].split('ƒ')
            identifier = fields[0]
            fieldcode=sfields[0][:4]
            entries=sfields[1:]
            if fieldcode=='039B' or fieldcode=='039E':
                for entry in entries:
                    entry = entry.replace("\n", "")
                    subcode=entry[0]
                    if subcode=='a':
                        typ=entry[1:]
                    if subcode=='c':
                        anno=entry[1:]
                test_input[fields[0]].append(f"{typ}: {anno}. ")

            if fieldcode=='150C' or fieldcode=='028C' or fieldcode=='028A':
                aut='_'
                exp=''
                aut_firstname=''
                code=''
                for entry in entries:
                    entry = entry.replace("\n", "")
                    subcode=entry[0]

                    if subcode=='a':
                        aut=entry[1:]

                    if subcode=='5':
                        aut_firstname=', '+entry[1:]
                    if subcode=='8':
                        exp=', '+entry[1:]
                    if subcode=='9':
                        code=entry[1:]
                    if fieldcode=='150C':
                        aut_type='auteur/primair'
                    elif fieldcode=='028A':
                        aut_type="hoofdauteur"
                    elif fieldcode=='028C':
                        aut_type='2e auteur'
                test_input[fields[0]].append(f"{aut_type}: {aut}{aut_firstname}{exp}. ")
            for entry in entries:
                entry = entry.replace("\n", "")
                subcode=entry[0]
                subfieldcode=fieldcode+subcode
                if subfieldcode in lookup_ppn.keys():
                    field_name=lookup_ppn[subfieldcode]
                    field_val=entry[1:]
                    test_input[fields[0]].append(f"{field_name}: {field_val}. ")
        if amount % 100000 == 0:
            print(f"Processed {amount} lines...")
        previous_fields = fields[0]


test_input = {k: ''.join(v) for k, v in test_input.items()}
print(len(test_input))
# Store data (serialize)
pickle.dump(test_input, open("lines_unseen.p", "wb"))  # save it into a file named save.p