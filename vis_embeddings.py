from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pickle
from sklearn.preprocessing import normalize

em = pickle.load(open("embeddings_couperus.p", "rb"))

X = np.array(list(em.values()), dtype=np.float32)
X_normalized = normalize(X)


kmeans = KMeans(n_clusters=15, random_state=42, n_init='auto')


        



cluster_labels = kmeans.fit_predict(X_normalized)

pca = PCA(n_components=2)
components = pca.fit_transform(X_normalized)

plt.scatter(
    components[:, 0],
    components[:, 1],
    c=cluster_labels,
    cmap="tab20",
    s=5
)
plt.show()
