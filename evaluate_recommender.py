
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"

# Load training and testing data
train = pd.read_csv(
    ARTIFACTS / "model_train_transactions.csv"
)

test = pd.read_csv(
    ARTIFACTS / "validation_transactions.csv"
)

courses = pd.read_csv(
    BASE / "Data" / "courses.csv"
)

clusters = pd.read_csv(
    ARTIFACTS / "validation_train_clusters.csv"
)

train_history = joblib.load(
    ARTIFACTS / "model_train_history.pkl"
)

popularity = joblib.load(
    ARTIFACTS / "validation_popularity_lookup.pkl"
)

# Course similarity based only on course metadata
course_features = pd.get_dummies(
    courses[["CourseCategory", "CourseLevel"]]
)

course_similarity = cosine_similarity(course_features)

course_ids = courses["CourseID"].tolist()

course_index = {
    course_id: i
    for i, course_id in enumerate(course_ids)
}

# Normalize course ratings
rating_lookup = {
    row.CourseID: (float(row.CourseRating) - 1) / 4
    for row in courses.itertuples()
}

cluster_lookup = dict(
    zip(clusters["UserID"], clusters["TrainCluster"])
)

# Prepare peer enrollment matrix from training only
training_users = list(train_history.keys())

history_matrix = np.array([
    [
        int(course_id in train_history[user_id])
        for course_id in course_ids
    ]
    for user_id in training_users
], dtype=float)

user_index = {
    user_id: i
    for i, user_id in enumerate(training_users)
}

def recommend(user_id, top_n=5):

    if user_id not in user_index:
        return []

    enrolled = train_history[user_id]
    cluster_id = cluster_lookup[user_id]

    # Find five similar learners
    similarities = cosine_similarity(
        history_matrix[user_index[user_id]].reshape(1, -1),
        history_matrix
    )[0]

    similarities[user_index[user_id]] = 0

    
    peer_indices = np.argsort(similarities)[::-1]

    peer_indices = [
        i for i in peer_indices
        if similarities[i] > 0
    ][:5]

    peer_total = sum(
        similarities[i] for i in peer_indices
    )

    enrolled_indices = sorted(
        course_index[cid]
        for cid in enrolled
        if cid in course_index
    )

    results = []

    for course_id in course_ids:

        if course_id in enrolled:
            continue


        idx = course_index[course_id]

        content_score = (
            course_similarity[idx, enrolled_indices].mean()
            if enrolled_indices else 0
        )

        popularity_score = popularity.get(
            (cluster_id, course_id), 0
        )

        peer_score = sum(
            similarities[i]
            for i in peer_indices
            if course_id in train_history[training_users[i]]
        )

        similar_learner_score = (
            peer_score / peer_total if peer_total > 0 else 0
        )

        rating_score = rating_lookup[course_id]

        final_score = (
            0.40 * content_score
            + 0.25 * popularity_score
            + 0.15 * rating_score
            + 0.20 * similar_learner_score
        )

        results.append((course_id, final_score))

    results.sort(key=lambda x: x[1], reverse=True)

    return [course_id for course_id, score in results[:top_n]]

print("\nEVALUATION RECOMMENDER READY")
print("Training learners:", len(training_users))
print("Available courses:", len(course_ids))

print("\nSample recommendations for U00003:")
print(recommend("U00003", top_n=5))

# STEP 9.9: Evaluate Recommendations

# Create future enrollment history from test data
test_history = (
    test.groupby("UserID")["CourseID"]
    .apply(set)
    .to_dict()
)

# Evaluate only learners present in both datasets
evaluation_users = sorted(
    set(train_history.keys()) &
    set(test_history.keys())
)

precision_scores = []
hit_scores = []
evaluation_results = []

for user_id in evaluation_users:

    recommended = recommend(user_id, top_n=5)
    actual_courses = test_history[user_id]

    # Count correctly predicted future enrollments
    hits = len(set(recommended) & actual_courses)

    precision = hits / 5
    hit = int(hits > 0)

    precision_scores.append(precision)
    hit_scores.append(hit)

    evaluation_results.append({
        "UserID": user_id,
        "RecommendedCourses": ",".join(recommended),
        "ActualFutureCourses": ",".join(
            sorted(actual_courses)
        ),
        "Hits": hits,
        "PrecisionAt5": precision,
        "HitAt5": hit
    })

# Calculate overall metrics
precision_at_5 = np.mean(precision_scores)
hit_rate_at_5 = np.mean(hit_scores)

print("\nRECOMMENDATION EVALUATION RESULTS")
print("Evaluated learners:", len(evaluation_users))
print(f"Precision@5: {precision_at_5:.4f}")
print(f"Hit Rate@5: {hit_rate_at_5:.4f}")
print("Learners with at least one hit:", sum(hit_scores))

# Save evaluation results
pd.DataFrame(evaluation_results).to_csv(
    ARTIFACTS / "recommendation_evaluation.csv",
    index=False
)

print("\nRecommendation evaluation saved successfully!")

# STEP 10: Global Popularity Baseline

popular_courses = (
    train.groupby("CourseID")
    .size()
    .sort_values(ascending=False)
    .index.tolist()
)

baseline_precisions = []
baseline_hits = []

for user_id in evaluation_users:

    enrolled = train_history[user_id]

    # Exclude courses already enrolled in
    recommended = [
        course_id
        for course_id in popular_courses
        if course_id not in enrolled
    ][:5]

    actual = test_history[user_id]

    hits = len(set(recommended) & actual)

    baseline_precisions.append(hits / 5)
    baseline_hits.append(int(hits > 0))

baseline_precision = np.mean(baseline_precisions)
baseline_hit_rate = np.mean(baseline_hits)

print("\nMODEL COMPARISON")

print(
    f"Hybrid Precision@5: {precision_at_5:.4f}"
)
print(
    f"Popularity Precision@5: {baseline_precision:.4f}"
)

print(
    f"Hybrid Hit Rate@5: {hit_rate_at_5:.4f}"
)
print(
    f"Popularity Hit Rate@5: {baseline_hit_rate:.4f}"
)
# STEP 11: Compare Recommendation Components

def evaluate_component(method):
    precisions = []
    hits_list = []

    for user_id in evaluation_users:
        enrolled = train_history[user_id]
        cluster_id = cluster_lookup[user_id]

        enrolled_indices = [
            course_index[cid]
            for cid in enrolled
            if cid in course_index
        ]

        scores = []

        for course_id in course_ids:
            if course_id in enrolled:
                continue

            idx = course_index[course_id]

            if method == "Content-Based":
                score = (
                    course_similarity[idx, enrolled_indices].mean()
                    if enrolled_indices else 0
                )

            elif method == "Cluster Popularity":
                score = popularity.get(
                    (cluster_id, course_id), 0
                )

            elif method == "Course Rating":
                score = rating_lookup[course_id]

            scores.append((course_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        recommended = [
            course_id for course_id, score in scores[:5]
        ]

        actual = test_history[user_id]
        hits = len(set(recommended) & actual)

        precisions.append(hits / 5)
        hits_list.append(int(hits > 0))

    return np.mean(precisions), np.mean(hits_list)


results = [
    {
        "Method": "Global Popularity",
        "Precision@5": baseline_precision,
        "HitRate@5": baseline_hit_rate
    },
    {
        "Method": "Hybrid",
        "Precision@5": precision_at_5,
        "HitRate@5": hit_rate_at_5
    }
]

for method in [
    "Content-Based",
    "Cluster Popularity",
    "Course Rating"
]:
    precision, hit_rate = evaluate_component(method)

    results.append({
        "Method": method,
        "Precision@5": precision,
        "HitRate@5": hit_rate
    })

comparison_df = pd.DataFrame(results)

print("\nFINAL RECOMMENDATION MODEL COMPARISON")
print(comparison_df.to_string(index=False))

comparison_df.to_csv(
    ARTIFACTS / "model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")




