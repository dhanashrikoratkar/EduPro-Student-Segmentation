
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

# Project paths
BASE = Path(__file__).resolve().parent
DATA = BASE / "Data"
ARTIFACTS = BASE / "artifacts"

# Load datasets
users = pd.read_csv(DATA / "users.csv")
courses = pd.read_csv(DATA / "courses.csv")
train = pd.read_csv(
    ARTIFACTS / "model_train_transactions.csv"
)

# Combine training transactions with course details
details = train.merge(
    courses,
    on="CourseID",
    how="left",
    validate="many_to_one"
)

# Build training-only learner features
profiles = []

for user_id, group in details.groupby("UserID"):
    category_counts = group["CourseCategory"].value_counts()
    level_counts = group["CourseLevel"].value_counts()

    beginner = (group["CourseLevel"] == "Beginner").sum()
    intermediate = (group["CourseLevel"] == "Intermediate").sum()
    advanced = (group["CourseLevel"] == "Advanced").sum()

    total = len(group)

    profiles.append({
        "UserID": user_id,
        "TotalCoursesEnrolled": total,
        "AverageCourseRating": group["CourseRating"].mean(),
        "AverageSpending": group["Amount"].mean(),
        "BeginnerEnrollments": beginner,
        "IntermediateEnrollments": intermediate,
        "AdvancedEnrollments": advanced,
        "DiversityScore": group["CourseCategory"].nunique(),
        "AverageCoursesPerCategory": total / max(
            group["CourseCategory"].nunique(), 1
        ),
        "EnrollmentFrequency": total / 10,
        "LearningDepthIndex": advanced / max(total, 1),
        "PreferredCategory": category_counts.idxmax(),
        "PreferredLevel": level_counts.idxmax()
    })

features = pd.DataFrame(profiles)

# Add demographic feature
features = features.merge(
    users[["UserID", "Age"]],
    on="UserID",
    how="left",
    validate="one_to_one"
)

# Mark learners with limited history
features["SparseEnrollmentFlag"] = (
    features["TotalCoursesEnrolled"] < 2
).astype(int)

features.loc[
    features["SparseEnrollmentFlag"] == 1,
    ["PreferredCategory", "PreferredLevel"]
] = "Unknown"

# Select model features
numeric_columns = [
    "TotalCoursesEnrolled",
    "AverageCourseRating",
    "AverageSpending",
    "BeginnerEnrollments",
    "IntermediateEnrollments",
    "AdvancedEnrollments",
    "DiversityScore",
    "AverageCoursesPerCategory",
    "EnrollmentFrequency",
    "LearningDepthIndex",
    "Age",
    "SparseEnrollmentFlag"
]

categorical_columns = [
    "PreferredCategory",
    "PreferredLevel"
]

# Apply training-only preprocessing
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_columns),
    ("categorical",
     OneHotEncoder(handle_unknown="ignore",
                   sparse_output=False),
     categorical_columns)
])

X_train = preprocessor.fit_transform(features)

# Train K-Means
model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

features["TrainCluster"] = model.fit_predict(X_train)

# Calculate validation score
score = silhouette_score(
    X_train,
    features["TrainCluster"],
    sample_size=min(2000, len(features)),
    random_state=42
)

# Save training-only results
features.to_csv(
    ARTIFACTS / "validation_train_clusters.csv",
    index=False
)

joblib.dump(
    {"model": model, "preprocessor": preprocessor},
    ARTIFACTS / "validation_clustering_model.pkl"
)

print("\nTRAINING-ONLY CLUSTERING RESULTS")
print("Learners:", len(features))
print("Feature matrix shape:", X_train.shape)
print("Silhouette Score:", round(score, 4))

print("\nCluster Distribution:")
print(features["TrainCluster"].value_counts())

print("\nTraining-only clustering saved successfully!")

# STEP 9.7: Rebuild Training-Only Cluster Popularity

# Connect training transactions to training-only clusters
train_cluster_transactions = train.merge(
    features[["UserID", "TrainCluster"]],
    on="UserID",
    how="inner",
    validate="many_to_one"
)

# Count enrollments for each course in each cluster
train_cluster_popularity = (
    train_cluster_transactions
    .groupby(["TrainCluster", "CourseID"])
    .size()
    .reset_index(name="EnrollmentCount")
)

# Normalize popularity within each cluster
train_cluster_popularity["PopularityScore"] = (
    train_cluster_popularity["EnrollmentCount"]
    /
    train_cluster_popularity.groupby("TrainCluster")[
        "EnrollmentCount"
    ].transform("sum")
)

# Create lookup dictionary
train_cluster_popularity_lookup = {
    (int(row.TrainCluster), row.CourseID): float(row.PopularityScore)
    for row in train_cluster_popularity.itertuples()
}

# Save results separately from the existing app
train_cluster_popularity.to_csv(
    ARTIFACTS / "validation_cluster_popularity.csv",
    index=False
)

joblib.dump(
    train_cluster_popularity_lookup,
    ARTIFACTS / "validation_popularity_lookup.pkl"
)

print("\nTRAINING-ONLY POPULARITY RESULTS")
print(
    "Total cluster-course combinations:",
    len(train_cluster_popularity)
)

print("\nTop 5 courses in Cluster 0:")
print(
    train_cluster_popularity[
        train_cluster_popularity["TrainCluster"] == 0
    ].sort_values(
        "EnrollmentCount",
        ascending=False
    ).head(5).to_string(index=False)
)

print("\nTraining-only popularity saved successfully!")
