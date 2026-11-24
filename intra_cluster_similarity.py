
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
features = pd.read_csv(
    "artifacts/processed_learner_features.csv"
)

clusters = pd.read_csv(
    "artifacts/clustered_learners.csv"
)

# Match learners by UserID
df = features.merge(
    clusters[["UserID", "Cluster"]],
    on="UserID",
    validate="one_to_one"
)

feature_columns = [
    col for col in features.columns
    if col != "UserID"
]

results = []

for cluster_id, group in df.groupby("Cluster"):

    X = group[feature_columns].to_numpy(dtype=float)
    n = len(X)

    # Normalize feature vectors for cosine similarity
    norms = np.linalg.norm(X, axis=1)
    valid = norms > 0
    X = X[valid]
    n = len(X)

    if n < 2:
        avg_similarity = np.nan
    else:
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Sum pairwise cosine similarities without
        # constructing a large similarity matrix
        vector_sum = X.sum(axis=0)
        total = np.dot(vector_sum, vector_sum) - n

        avg_similarity = total / (n * (n - 1))

    results.append({
        "Cluster": cluster_id,
        "Learners": len(group),
        "AverageCosineSimilarity": avg_similarity
    })

results_df = pd.DataFrame(results)

print("\nINTRA-CLUSTER SIMILARITY RESULTS")
print(results_df.to_string(index=False))

results_df.to_csv(
    "artifacts/intra_cluster_similarity.csv",
    index=False
)

print("\nIntra-cluster similarity saved successfully!")
