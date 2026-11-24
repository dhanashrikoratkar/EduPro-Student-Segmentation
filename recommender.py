import joblib
import pandas as pd

# Load saved EduPro artifacts
artifacts = joblib.load("edupro_recommender_artifacts.pkl")

courses = artifacts["courses"].reset_index(drop=True)
learner_features = artifacts["learner_features"]
course_similarity = artifacts["course_similarity"]
learner_course_history = artifacts["learner_course_history"]
popularity_lookup = artifacts["popularity_lookup"]
rating_lookup = artifacts["rating_lookup"]


def recommend_courses(user_id, top_n=5):

    learner = learner_features[
        learner_features["UserID"] == user_id
    ]

    if learner.empty:
        return pd.DataFrame()

    cluster_id = learner.iloc[0]["Cluster"]

    enrolled_courses = learner_course_history.get(
        user_id, set()
    )

    course_indices = {
        course_id: index
        for index, course_id in enumerate(courses["CourseID"])
    }

    enrolled_indices = [
        course_indices[course_id]
        for course_id in enrolled_courses
        if course_id in course_indices
    ]
    # Calculate similarity between learners
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_user_ids = learner_features.loc[
        (learner_features["TotalCoursesEnrolled"] >= 2) |
        (learner_features["UserID"] == user_id),
        "UserID"
    ].tolist()

    history_matrix = np.array([
        [
            1 if cid in learner_course_history.get(uid, set()) else 0
            for cid in courses["CourseID"]
        ]
        for uid in similarity_user_ids
    ], dtype=float)

    selected_index = similarity_user_ids.index(user_id)

    similarity_scores = cosine_similarity(
        history_matrix[selected_index].reshape(1, -1),
        history_matrix
    )[0]

    similarity_scores[selected_index] = 0

    top_indices = np.argsort(similarity_scores)[::-1]

    similar_user_ids = [
        similarity_user_ids[i]
        for i in top_indices
        if similarity_scores[i] > 0
    ][:5]

    total_peer_similarity = sum(
        similarity_scores[i] for i in top_indices
        if similarity_user_ids[i] in similar_user_ids
        and similarity_scores[i] > 0
    )
    recommendations = []

    for course_index, course in courses.iterrows():

        course_id = course["CourseID"]

        if course_id in enrolled_courses:
            continue

        if enrolled_indices:
            content_score = course_similarity[
                course_index, enrolled_indices
            ].mean()
        else:
            content_score = 0

        popularity_score = popularity_lookup.get(
            (cluster_id, course_id), 0
        )
        peer_score = sum(
            similarity_scores[i]
            for i, peer_id in enumerate(similarity_user_ids)
            if peer_id in similar_user_ids
            and course_id in learner_course_history.get(peer_id, set())
        )

        similar_learner_score = (
            peer_score / total_peer_similarity
            if total_peer_similarity > 0 else 0
        )
        rating_score = rating_lookup.get(course_id, 0)

        final_score = (
             0.40 * content_score
             + 0.25 * popularity_score
             + 0.15 * rating_score
             + 0.20 * similar_learner_score
        )

        recommendations.append({
            "CourseID": course_id,
            "CourseName": course["CourseName"],
            "CourseCategory": course["CourseCategory"],
            "CourseLevel": course["CourseLevel"],
            "CourseRating": course["CourseRating"],
            "ContentSimilarity": content_score,
            "ClusterPopularity": popularity_score,
            "RatingScore": rating_score,
            "SimilarLearnerScore": similar_learner_score,
            "FinalScore": final_score
        })

    result = pd.DataFrame(recommendations)

    return result.sort_values(
        "FinalScore", ascending=False
    ).head(top_n).reset_index(drop=True)

print("\nSaved artifact components:")
print(list(artifacts.keys()))

print("\nOld cluster distribution:")
print(learner_features["Cluster"].value_counts())

# Load newly trained clusters
new_learners = pd.read_csv(
    "artifacts/clustered_learners.csv"
)

# Compare cluster assignments by UserID
comparison = learner_features[
    ["UserID", "Cluster"]
].merge(
    new_learners[["UserID", "Cluster"]],
    on="UserID",
    suffixes=("_Old", "_New")
)

print("\nLearners compared:", len(comparison))

print(
    "Matching cluster assignments:",
    (comparison["Cluster_Old"] ==
     comparison["Cluster_New"]).sum()
)

# STEP 8.4: Test hybrid recommendations

print("\nPERSONALIZED RECOMMENDATIONS")

test_user = "U00003"

recommendations = recommend_courses(
    test_user,
    top_n=5
)

print("\nRecommendations for:", test_user)

print(
    recommendations[
        [
            "CourseName",
            "CourseCategory",
            "FinalScore"
        ]
    ].to_string(index=False)
)

# STEP 8.5: Recommendation Validation

enrolled = learner_course_history.get(
    test_user, set()
)

recommended = set(
    recommendations["CourseID"]
)

duplicates = enrolled.intersection(recommended)

print("\nRECOMMENDATION VALIDATION")

print("Learner ID:", test_user)
print("Previously enrolled courses:", len(enrolled))
print("Recommended courses:", len(recommended))
print("Already enrolled courses recommended:", len(duplicates))

if len(duplicates) == 0:
    print("PASS: No previously enrolled courses recommended!")
else:
    print("FAIL: Previously enrolled courses found!")

# STEP 9.1: Check transaction dates

transactions = pd.read_csv(
    "Data/transactions.csv"
)

print("\nTRANSACTION DATASET")

print("Total transactions:", len(transactions))

print("\nColumn names:")
print(transactions.columns.tolist())

print("\nFirst five records:")
print(transactions.head().to_string(index=False))

# STEP 9.2: Chronological Train-Test Split

# Convert dates into datetime format
transactions["TransactionDate"] = pd.to_datetime(
    transactions["TransactionDate"],
    dayfirst=True
)

# Sort transactions by date
transactions = transactions.sort_values(
    "TransactionDate"
).reset_index(drop=True)

# Use a strict date cutoff
cutoff_date = pd.Timestamp("2025-10-18")

train_transactions = transactions[
    transactions["TransactionDate"] < cutoff_date
].copy()

test_transactions = transactions[
    transactions["TransactionDate"] >= cutoff_date
].copy()

# Display results
print("\nTRAIN-TEST SPLIT RESULTS")

print("Training transactions:", len(train_transactions))
print("Testing transactions:", len(test_transactions))

print(
    "Training period:",
    train_transactions["TransactionDate"].min(),
    "to",
    train_transactions["TransactionDate"].max()
)

print(
    "Testing period:",
    test_transactions["TransactionDate"].min(),
    "to",
    test_transactions["TransactionDate"].max()
)

# Save datasets
train_transactions.to_csv(
    "artifacts/train_transactions.csv",
    index=False
)

test_transactions.to_csv(
    "artifacts/test_transactions.csv",
    index=False
)

print("\nTraining and testing datasets saved!")

# STEP 9.4: Build Training-Only Learner History

train_history = (
    train_transactions.groupby("UserID")["CourseID"]
    .apply(set)
    .to_dict()
)

print("\nTRAINING-ONLY LEARNER HISTORY")

print("Learners with training history:", len(train_history))

print(
    "Training courses for U00003:",
    train_history.get("U00003", set())
)

# Save training history
joblib.dump(
    train_history,
    "artifacts/train_learner_history.pkl"
)

print("\nTraining learner history saved successfully!")

# STEP 9.5: Training-Only Cluster Popularity

# Load learner clusters
cluster_data = pd.read_csv(
    "artifacts/clustered_learners.csv"
)

# Join training transactions with clusters
train_with_clusters = train_transactions.merge(
    cluster_data[["UserID", "Cluster"]],
    on="UserID",
    how="inner"
)

# Count course enrollments within each cluster
cluster_popularity = (
    train_with_clusters
    .groupby(["Cluster", "CourseID"])
    .size()
    .reset_index(name="EnrollmentCount")
)

# Normalize popularity within each cluster
cluster_popularity["PopularityScore"] = (
    cluster_popularity["EnrollmentCount"] /
    cluster_popularity.groupby("Cluster")[
        "EnrollmentCount"
    ].transform("sum")
)

# Create popularity lookup
train_popularity_lookup = {
    (row.Cluster, row.CourseID): row.PopularityScore
    for row in cluster_popularity.itertuples()
}

# Save results
cluster_popularity.to_csv(
    "artifacts/train_cluster_popularity.csv",
    index=False
)

joblib.dump(
    train_popularity_lookup,
    "artifacts/train_popularity_lookup.pkl"
)

print("\nTRAINING-ONLY CLUSTER POPULARITY")

print("Total cluster-course combinations:", len(cluster_popularity))

print("\nTop 5 courses in Cluster 0:")
print(
    cluster_popularity[
        cluster_popularity["Cluster"] == 0
    ].sort_values(
        "EnrollmentCount",
        ascending=False
    ).head(5).to_string(index=False)
)

print("\nTraining cluster popularity saved successfully!")

