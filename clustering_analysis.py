
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load preprocessed learner data
data_path = Path("artifacts/processed_learner_features.csv")
df = pd.read_csv(data_path)

# Use numeric features for clustering
X = df.select_dtypes(include="number")

inertia = []
silhouette_scores = []

# Test cluster counts from 2 to 10
for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    inertia.append(model.inertia_)

    score = silhouette_score(
        X,
        labels,
        sample_size=min(2000, len(X)),
        random_state=42
    )
    silhouette_scores.append(score)

    print(f"K={k}, Silhouette Score={score:.4f}")

# Create elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.savefig("artifacts/elbow_plot.png")
plt.show()

# Create silhouette graph
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.grid(True)
plt.savefig("artifacts/silhouette_plot.png")
plt.show()

print("Clustering analysis completed successfully!")

import pickle

# Train final K-Means model
final_model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

cluster_labels = final_model.fit_predict(X)

# Load original learner information
learners = pd.read_csv("learner_features.csv")

# Add cluster labels
learners["Cluster"] = cluster_labels

# Save clustered learners
learners.to_csv(
    "artifacts/clustered_learners.csv",
    index=False
)

# Save trained model
with open("artifacts/kmeans_model.pkl", "wb") as file:
    pickle.dump(final_model, file)

print("\nFinal K-Means model trained successfully!")
print("\nLearners in each cluster:")
print(learners["Cluster"].value_counts())
print("\nSaved clustered_learners.csv and kmeans_model.pkl")

# STEP 5: Cluster Profiling

profile_columns = [
    "TotalCoursesEnrolled",
    "AverageCourseRating",
    "AverageSpending",
    "EnrollmentFrequency",
    "LearningDepthIndex",
    "DiversityScore"
]

cluster_profile = learners.groupby("Cluster")[
    profile_columns
].mean().round(2)

print("\nCLUSTER PROFILES:")
print(cluster_profile.to_string())

# Save cluster profiles
cluster_profile.to_csv(
    "artifacts/cluster_profiles.csv"
)

print("\nCluster profiles saved successfully!")

# STEP 6: Add meaningful cluster names

cluster_names = {
    0: "Occasional Learners",
    1: "Highly Engaged Learners"
}

learners["ClusterName"] = learners["Cluster"].map(
    cluster_names
)

# Save updated learner dataset
learners.to_csv(
    "artifacts/clustered_learners.csv",
    index=False
)

print("\nCluster names assigned successfully!")
print(
    learners["ClusterName"].value_counts()
)

# STEP 7: Hierarchical Clustering Validation

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# Train hierarchical clustering with 2 clusters
hierarchical_model = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)

hierarchical_labels = hierarchical_model.fit_predict(X)

# Calculate silhouette score
hierarchical_score = silhouette_score(
    X,
    hierarchical_labels,
    sample_size=min(2000, len(X)),
    random_state=42
)

# Compare with K-Means labels
agreement_score = adjusted_rand_score(
    learners["Cluster"],
    hierarchical_labels
)

print("\nHIERARCHICAL CLUSTERING RESULTS:")
print(f"Silhouette Score: {hierarchical_score:.4f}")
print(f"Agreement with K-Means (ARI): {agreement_score:.4f}")

print("\nHierarchical Cluster Distribution:")
print(pd.Series(hierarchical_labels).value_counts())

# Save results
learners["HierarchicalCluster"] = hierarchical_labels

learners.to_csv(
    "artifacts/clustering_validation.csv",
    index=False
)

print("\nHierarchical clustering validation saved!")
