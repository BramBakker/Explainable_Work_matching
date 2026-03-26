from collections import defaultdict
import pickle
from collections import Counter
# Work group ID store (global)
work_groups = {}
# Expression group ID store scoped to work
expr_groups_within_work = defaultdict(dict)

triples = []
test_triples = []
def get_group_id(section_str, group_dict, prefix):
    key = tuple(sorted(section_str))
    if key not in group_dict:
        group_dict[key] = f"{prefix}_{len(group_dict)+1}"
    return group_dict[key]

aut_counts={}

#top_aut_list=['068435061#couperus',"072614730#bruna","068611447#hulst","068243391#multatuli",
#          "068808658#tolstoj","068690436#kieviet", "068291965#simenon", 
#          "069804354#beckman","068405944#balzac","074333682#slee","069815100#teng","068412606#terlouw",
#          "068800355#thijssing-boer","068256671#reve","07492866X#murakami","075153416#grunberg", ]
top_aut_list=["072614730#bruna"]

unseen_aut_list=['069410488#brusselmans',
'069922047#moeyaert',
'068459823#gulik',
'068842368#wolkers',
'06936351X#vanderspoel',
'307884864#ivanov',
'068588542#maclean',
'068369123#defoe',
'068242964#twain',
'141105798#baldacci',
'069031584#verleyen',
'068256671#reve',
'069088330#ludlum',
'069670587#spyri',
'06898975X#wallace'
]
with open("romans_kinderboeken_CAP_nieuw.txt", "r", encoding="utf-8") as f:
    next(f) 

    for line in f:
        if not line:
            continue

        parts = line.split("\t")
        if len(parts)>1:
            identifier = parts[0]
            data = parts[1]
            tokens = data.split("\\\\")
            work_data = []
            expr_data = []
            if tokens[0] in aut_counts:
                aut_counts[tokens[0]] += 1
            else:
                aut_counts[tokens[0]] = 1
                
            for token in tokens:
                if token.startswith("W3"):
                    title_token=token
                if token.startswith("W"):
                    work_data.append(token)
                elif token.startswith("E"):
                    expr_data.append(token)
            work_string=" ".join(work_data)
            # Assign work group
            work_id = get_group_id(work_string, work_groups, "W")


            # Assign expression group scoped *within* this work
            if any(substring in tokens[0] for substring in unseen_aut_list):
                triples.append((identifier, work_id))
            #triples.append((identifier, work_id))
print(len(triples))
pickle.dump(triples, open("id_pairs_unseen.p", "wb"))  # save it into a file named save.p
