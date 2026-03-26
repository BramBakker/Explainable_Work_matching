import random
import pickle
import Levenshtein


input = pickle.load(open("lines_full.p", "rb"))
targets = pickle.load(open("id_pairs_full.p", "rb"))
x = []
y = []

for trip in targets:
    if trip[0] in input.keys():
        cor_input = input[trip[0]]
        x.append(cor_input)
        y.append(trip[1])   # now only work_id remains meaningful

# Build index_by_work only
index_by_work = {}
for i, work_id in enumerate(y):
    index_by_work.setdefault(work_id, []).append(i)

with open("new_pairs.txt", "w", encoding="utf-8") as f_work:
    print(len(x))
    for i in range(0, len(x)):
        anchor_text = x[i]
        anchor_work = y[i]
        #if "ondertitel" in anchor_text:
        #    anchor_subtitle = anchor_text.split("ondertitel: ", 1)[1].split(".", 1)[0]
        #if "titel" in anchor_text:
        #    anchor_title = anchor_text.split("titel: ", 1)[1].split(".", 1)[0]
        #if "hoofdauteur" in anchor_text:
        #    anchor_author = anchor_text.split("hoofdauteur: ", 1)[1].split(".", 1)[0]
        same_work = index_by_work[anchor_work]

        # Positive: same work, not self
        pos_candidates = [j for j in same_work if j != i]

        # Negative: different work
        neg_work_candidates = [j for j in range(i,i+50) if y[j] != anchor_work]

        # Write to neg_work_triplets.txt
        if pos_candidates and pos_candidates:
        #    real_neg_candidates=[]
        #    real_pos_candidates=[]
        #    for p in pos_candidates:
        #        potential_text=x[p]
        #        if "titel: " in potential_text:
        #            if "hoofdauteur" in anchor_text:
        #                if "hoofdauteur" in potential_text:
        ##                    potential_author = potential_text.split("hoofdauteur: ", 1)[1].split(".", 1)[0]
         #                   if potential_author==anchor_author:
         #                       potential_title = potential_text.split("titel: ", 1)[1].split(".", 1)[0]
        #                        dist_calc = Levenshtein.distance(potential_title, anchor_title)
        #                        if len(potential_title)<10:
        #                            if potential_title!=anchor_title:
        #                                if "vertaling" not in anchor_text:
        #                                    if "vertaling" not in potential_text:
        #                                        f_work.write(f"{anchor_text}\t{potential_text}\t0\n")
                    #pot_lang=potential_text.split("COL taal VAL ", 1)[1].split("COL", 1)[0]
                #if pot_lang!=anchor_lang:
                    #if  "vert. van" not in anchor_text and "vert. van" not in potential_text:
                        #print(potential_text)
                        #print(anchor_text)
            neg_work_idx = random.choice(neg_work_candidates)
            pos_work_idx = random.choice(pos_candidates)

        #Positive pair
        f_work.write(f"{anchor_text}\t{x[neg_work_idx]}\t1\n")
        f_work.write(f"{anchor_text}\t{x[pos_work_idx]}\t0\n")

        if i % 10000 == 0:
            print(f"Processed {i} items...")