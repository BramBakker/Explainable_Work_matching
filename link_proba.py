import pickle
import numpy as np
from sklearn.preprocessing import normalize
import torch
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein

em = pickle.load(open("embeddings_unseen_newest.p", "rb"))
texts = pickle.load(open("lines_unseen.p", "rb"))
X = torch.stack(list(em.values())).detach().cpu().numpy().astype(np.float32)
X_normalized = normalize(X)
sim_matrix = cosine_similarity(X_normalized,X_normalized)
ids_list = list(em.keys())

links=[]
for i in range(len(sim_matrix)):
    for j in range(len(sim_matrix)):
        if i!=j:
            if sim_matrix[i][j]>0.95:
                links.append((i,j))
example_counter=0
already_done=[]
example_pairs=[]
bruna=0

hard_pairs=[]

for link in links:
    id_1=ids_list[link[0]]
    id_2=ids_list[link[1]]
    anchor_text=texts[id_1]
    potential_text=texts[id_2]
    #if "Bruna" in anchor_text:
     #   bruna+=1
     #   if id_1 not in already_done and bruna<10:
    #        example_pairs.append([anchor_text,potential_text])

#    anchor_t = anchor_text.split("titel: ", 1)[1].split(".", 1)[0]
#    anchor_title=anchor_t.replace("@","")
#    potential_t=potential_text.split("titel: ", 1)[1].split(".", 1)[0]
#    potential_title=potential_t.replace("@","")
#    if potential_title!= anchor_title:
#        if "vertaling van" in potential_text:
#            pot_tr = potential_text.split("vertaling van: ", 1)[1].split(".", 1)[0]
#            if pot_tr!=anchor_title:
#                dist_calc = Levenshtein.distance(pot_tr, anchor_title)
#                if dist_calc>3:
#                    example_counter+=1
#                    print(texts[id_1])
#                    print("with probability"+str(sim_matrix[link[0]][link[1]]))
#                    print(texts[id_2])
#                    print("----------------------------------------------")
    if id_1 not in already_done:
        anchor_t = anchor_text.split("titel: ", 1)[1].split(".", 1)[0]
        anchor_title=anchor_t.replace("@","")
        title_words=anchor_title.split(" ")
        if "vertaling van" in anchor_text:
            anchor_tr = anchor_text.split("vertaling van: ", 1)[1].split(".", 1)[0]
            translat=True
        else:
            translat=False
        potential_t=potential_text.split("titel: ", 1)[1].split(".", 1)[0]
        potential_title=potential_t.replace("@","")
        print(anchor_text)
        print(potential_text)

        print("with probability"+str(sim_matrix[link[0]][link[1]]))

        print("----------------------------------------------")


        if potential_title!= anchor_title:
            if len(title_words)>1:
                if "vertaling van" in potential_text:
                    pot_tr = potential_text.split("vertaling van: ", 1)[1].split(".", 1)[0]
                    if pot_tr!=anchor_title:
                        if translat==True:
                            if pot_tr!= anchor_tr:
                                continue
                            elif anchor_tr!= potential_title:
                                continue
                            else:
                                example_pairs.append([anchor_text,potential_text])
                                print(anchor_text)
                                print(potential_text)

                                print("with probability"+str(sim_matrix[link[0]][link[1]]))

                                print("----------------------------------------------")
                                already_done.append(id_1)

                        else:
                            example_pairs.append([anchor_text,potential_text])
                            print(anchor_text)
                            print(potential_text)

                            print("with probability"+str(sim_matrix[link[0]][link[1]]))

                            print("----------------------------------------------")
                            already_done.append(id_1)


                else:
                    example_pairs.append([anchor_text,potential_text])
                    print(anchor_text)
                    print(potential_text)
                    print("with probability"+str(sim_matrix[link[0]][link[1]]))

                    print("----------------------------------------------")

                    already_done.append(id_1)


print(len(already_done))
#        else:
#            example_counter+=1
#            print(texts[id_1])
#            print("with probability"+str(sim_matrix[link[0]][link[1]]))
#            print(texts[id_2])
#            print("----------------------------------------------")
pickle.dump(example_pairs, open("example_pairs.p", "wb")) 