from sklearn.cluster import KMeans
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from itertools import product
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
em = pickle.load(open("embeddings.p", "rb"))
texts = pickle.load(open("lines.p", "rb"))
pairs=pickle.load(open("id_pairs.p", "rb"))
w_dict={}
for pair in pairs:
    if pair[1] not in w_dict.keys():
        w_dict[pair[1]]=[pair[0]]
    else:
        w_dict[pair[1]].append(pair[0])
print(len(w_dict))
X = np.array(list(em.values()), dtype=np.float32)
X_normalized = normalize(X)


kmeans = KMeans(n_clusters=150, random_state=42, n_init='auto')


        



cluster_labels = kmeans.fit_predict(X_normalized)
print(cluster_labels)
ids = list(em.keys())
distances=cdist(X_normalized, X_normalized, 'euclidean')
#for d in range(len(distances)):
#    for s in range(len(distances[d])):
#        if distances[d][s]<0.6:
#            obj_id = ids[s]
            
#            print(texts[obj_id])
#    print('------------------------------------')

clustered_texts = {}
clustered_ids = {}
for i, cluster_id in enumerate(cluster_labels):
    if cluster_id not in clustered_texts:
        clustered_texts[cluster_id] = []
        clustered_ids[cluster_id] = []
    
    obj_id = ids[i]
    text_content = texts[obj_id]
    clustered_texts[cluster_id].append(text_content)
    clustered_ids[cluster_id].append(obj_id)

#for i in range(len(clustered_texts)):
#    print("--------------------")
#    for t in clustered_texts[i]:
#        print(f"- {t}")
clusters_a = [set(v) for v in w_dict.values()]
clusters_b = [set(v) for v in clustered_ids.values()]
for cluster_b in clusters_b:
    for id in cluster_b:
        if "Bruna" in texts[id]:
            if "woord" in texts[id]:
                print(texts[id])
    print("-------------------")



for cluster_b in clusters_b:
    for cluster_a in clusters_a:
        overlap =cluster_a&cluster_b
        if len(overlap)>1:
                if len(cluster_b)>len(cluster_a):
                    for id in cluster_a:
                        if "Bruna" in texts[id]:
                            print(texts[id])
                    print("----------------compared to our cluster------------------")
                    for id in cluster_b:
                        if "Bruna" in texts[id]:

                            print(texts[id])
                    print("-----------------------------------------------------------")