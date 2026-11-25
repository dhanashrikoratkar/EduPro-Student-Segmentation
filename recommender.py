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
