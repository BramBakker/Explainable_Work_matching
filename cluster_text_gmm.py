from sklearn.cluster import KMeans
import numpy as np
import pickle
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture

em = pickle.load(open("embeddings_couperus.p", "rb"))
texts = pickle.load(open("lines_couperus.p", "rb"))
X = np.array(list(em.values()), dtype=np.float32)
X_normalized = normalize(X)

gmm = GaussianMixture(
    n_components=10, 
    covariance_type='diag', 
    reg_covar=0.01,  # Increase from 1e-6 to 1e-2 or even 0.1
    random_state=42
)
gmm.fit(X_normalized)
probs=gmm.predict_proba(X_normalized)
cluster_labels = probs.argmax(axis=1)
# 2. Calculate the Margin
# Sort each row to get the two highest probabilities
sorted_probs = np.sort(probs, axis=1)
top_1 = sorted_probs[:, -1]
top_2 = sorted_probs[:, -2]
margin = top_1 - top_2

# 3. Calculate Log-Likelihood (to identify and filter loners)
log_likelihood = gmm.score_samples(X_normalized)
ambiguity_threshold = 0.95  # Adjust based on your data
loner_threshold = np.percentile(log_likelihood, 20) # Filter bottom 10% of loners

ambiguous_indices = np.where(
    (margin < ambiguity_threshold) & 
    (log_likelihood > loner_threshold)
)[0]


# 1. Create a list of the keys in the same order as X was created
ids_list = list(em.keys())

print(f"\n--- Found {len(ambiguous_indices)} Ambiguous Nodes ---")
print(f"Criteria: Margin < {ambiguity_threshold} and Log-Likelihood > {loner_threshold:.2f}\n")
clustered_texts = {}

for i, cluster_id in enumerate(cluster_labels):
    if cluster_id not in clustered_texts:
        clustered_texts[cluster_id] = []
    
    obj_id = ids_list[i]
    text_content = texts[obj_id]
    clustered_texts[cluster_id].append(text_content)
for idx in ambiguous_indices:
    # Get the original ID
    record_id = ids_list[idx]
    
    # Retrieve the text and relevant metrics
    record_text = texts.get(record_id, "Text not found")
    node_margin = margin[idx]
    node_ll = log_likelihood[idx]
    
    # Optional: Get the IDs of the top 2 clusters to see the conflict
    best_clusters = np.argsort(probs[idx])[-2:][::-1]

    for t in clustered_texts[best_clusters[0]]:
        print(f"- {t}")
    print("orrr is it:")
    for t in clustered_texts[best_clusters[1]]:
        print(f"- {t}")

    print(f"ID: {record_id}")
    print(f"Margin: {node_margin:.4f} | Log-Likelihood: {node_ll:.2f}")
    print(f"Top 2 Clusters: {best_clusters[0]} ({probs[idx][best_clusters[0]]:.2%}) vs {best_clusters[1]} ({probs[idx][best_clusters[1]]:.2%})")
    print(f"Text: {record_text}")
    print("-" * 50)