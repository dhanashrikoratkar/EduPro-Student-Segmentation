
import streamlit as st
import pandas as pd
import joblib
from recommender import recommend_courses

# Page configuration
st.set_page_config(
    page_title="EduPro - Learner Segmentation",
    page_icon="📚",
    layout="wide"
)

st.html("""
<style>
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #6366F1;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    min-height: 110px;
}

[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: #64748B;
}

[data-testid="stMetricValue"] {
    font-weight: 700;
    color: #172554;
}

[data-testid="stMetric"]:hover {
    border-color: #6366F1;
    box-shadow: 0 6px 15px rgba(99,102,241,0.15);
}
</style>
""")


# Load saved artifacts
@st.cache_resource
def load_artifacts():
    return joblib.load("edupro_recommender_artifacts.pkl")

artifacts = load_artifacts()

learner_features = artifacts['learner_features']
courses = artifacts["courses"]

# Dashboard title


st.html("""
<div style="
    background: linear-gradient(135deg, #172554, #6366F1);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 25px;
">
<h1 style="color: white; font-size: 48px; font-weight: 900; margin: 0;">
EduPro
</h1>
<h2 style="color: white; font-size: 26px; font-weight: 800; margin: 10px 0;">
Student Segmentation and Course Recommendation
</h2>
<p style="color: #E0E7FF; font-size: 16px; margin: 0;">
Machine Learning | Personalized Learning | Data Analytics
</p>
</div>
""")

st.html("""
<style>

/* Section headings */
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3 {
    color: #172554;
    font-weight: 750;
    border-bottom: 2px solid #E0E7FF;
    padding-bottom: 10px;
}

/* Charts and tables */
[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"],
[data-testid="stPlotlyChart"],
[data-testid="stArrowVegaLiteChart"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

/* Selection fields */
[data-testid="stSelectbox"],
[data-testid="stMultiSelect"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 15px;
    border-radius: 12px;
}

/* Information and result boxes */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Buttons */
[data-testid="stButton"] button {
    border-radius: 10px;
    font-weight: 600;
}

</style>
""")




# Sidebar navigation

st.sidebar.title("📚 EduPro")

st.html("""
<style>
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 12px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: #263B70;
    padding: 14px;
    border-radius: 10px;
    width: 100%;
    transition: 0.2s;
    cursor: pointer;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: #4338CA;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background-color: #6366F1;
    font-weight: bold;
}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}
</style>
""")



page = st.sidebar.pills(
    "DASHBOARD NAVIGATION",
    [
        "Learner Profile Explorer",
        "Cluster Dashboard",
        "Personalized Recommendations",
        "Segment Comparison",
        "Personalized Learning Path"
    ],
    format_func=lambda x: {
        "Learner Profile Explorer": "👤 Learner Profile Explorer",
        "Cluster Dashboard": "📊 Cluster Dashboard",
        "Personalized Recommendations": "🎯 Personalized Recommendations",
        "Segment Comparison": "📈 Segment Comparison",
        "Personalized Learning Path": "🎓 Personalized Learning Path"
    }[x],
    selection_mode="single",
    default="Learner Profile Explorer"
)

if page is None:
    page = "Learner Profile Explorer"




st.subheader(page)

if page == "Learner Profile Explorer":
    st.write(
        "Select a learner to explore their profile, "
        "learning behavior, and assigned segment."
    )

    # Learner selection
    user_ids = sorted(learner_features['UserID'].unique())

    selected_user = st.selectbox(
        "Select Learner ID",
        user_ids
    )

    # Get selected learner
    learner = learner_features[
        learner_features['UserID'] == selected_user
    ].iloc[0]

    # Cluster labels
    cluster_names = {
        0: "Occasional Learners",
        1: "Highly Engaged, Diverse Learners"
    }

    cluster_name = cluster_names.get(
        learner['Cluster'], "Unknown Segment"
    )

    # Learner profile
    st.markdown("### Learner Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Learner ID", selected_user)
    col2.metric("Age", int(learner['Age']))
    col3.metric("Gender", learner['Gender'])

    st.markdown("### Assigned Learner Segment")

    st.success(
        f"Cluster {int(learner['Cluster'])}: {cluster_name}"
    )


    
    # Learning behavior metrics
    st.markdown("### 📊 Learning Behavior")

    col1, col2 = st.columns(2, gap="medium")

    col1.metric(
        "📚 Courses Enrolled",
        int(learner['TotalCoursesEnrolled'])
    )

    col2.metric(
        "💰 Average Spending",
        f"{learner['AverageSpending']:.2f}"
    )

    col3, col4 = st.columns(2, gap="medium")

    col3.metric(
        "⭐ Average Course Rating",
        f"{learner['AverageCourseRating']:.2f}"
    )

    col4.metric(
        "📂 Category Diversity",
        int(learner['DiversityScore'])
    )


    
    # Additional learner preferences
    st.markdown("### 🎯 Learning Preferences")

    col1, col2, col3 = st.columns(3, gap="medium")

    col1.metric(
        "📂 Preferred Category",
        learner['PreferredCategory']
    )

    col2.metric(
        "🎓 Preferred Level",
        learner['PreferredLevel']
    )

    col3.metric(
        "📊 Learning Depth Index",
        f"{learner['LearningDepthIndex']:.2f}"
    )



elif page == "Cluster Dashboard":

    st.markdown("## 📊 Cluster Dashboard")

    cluster_counts = (
        learner_features["Cluster"]
        .value_counts()
        .sort_index()
    )

    total_learners = int(cluster_counts.sum())

    # Summary metric cards
    col1, col2, col3 = st.columns(3, gap="medium")

    col1.metric(
        "👥 Total Learners",
        f"{total_learners:,}"
    )

    col2.metric(
        "📚 Occasional Learners",
        f"{cluster_counts.get(0, 0):,}"
    )

    col3.metric(
        "🎯 Highly Engaged Learners",
        f"{cluster_counts.get(1, 0):,}"
    )

    st.divider()

    # Cluster summary table
    cluster_summary = pd.DataFrame({
        "Cluster": cluster_counts.index,
        "Number of Learners": cluster_counts.values,
        "Percentage (%)": (
            cluster_counts.values / total_learners * 100
        ).round(2)
    })

    cluster_summary["Cluster"] = cluster_summary["Cluster"].map({
        0: "Occasional Learners",
        1: "Highly Engaged Learners"
    })

    # Side-by-side layout
    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown("### 📋 Cluster Summary")

        st.dataframe(
            cluster_summary,
            hide_index=True,
            width="stretch"
        )

    with right:
        st.markdown("### 📈 Cluster Distribution")

        st.bar_chart(
            cluster_summary.set_index("Cluster")[
                "Number of Learners"
            ],
            color="#6366F1"
        )

    st.divider()

    st.markdown("### 💡 Cluster Insights")

    st.info(
        "Occasional learners enroll in fewer courses, "
        "while highly engaged learners explore more "
        "courses across different categories."
    )



elif page == "Personalized Recommendations":

    st.markdown("## 🎯 Personalized Course Recommendations")

    st.caption(
        "Discover suitable courses based on individual "
        "learning behavior and preferences."
    )

    # Learner selection
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        selected_user = st.selectbox(
            "👤 Select Learner ID",
            sorted(learner_features["UserID"].unique()),
            key="recommendation_user"
        )

    with col2:
        top_n = st.slider(
            "🔢 Number of Recommendations",
            min_value=1,
            max_value=10,
            value=5
        )

    # Category and level filters
    col3, col4 = st.columns(2, gap="medium")

    with col3:
        category_filter = st.selectbox(
            "📂 Course Category",
            ["All"] + sorted(
                courses["CourseCategory"]
                .dropna().unique().tolist()
            )
        )

    with col4:
        level_filter = st.selectbox(
            "🎓 Course Level",
            ["All"] + sorted(
                courses["CourseLevel"]
                .dropna().unique().tolist()
            )
        )

    st.divider()

    # Generate recommendations
    recommendations = recommend_courses(
        selected_user,
        top_n=len(courses)
    )

    # Apply selected filters
    if category_filter != "All":
        recommendations = recommendations[
            recommendations["CourseCategory"] == category_filter
        ]

    if level_filter != "All":
        recommendations = recommendations[
            recommendations["CourseLevel"] == level_filter
        ]

    recommendations = recommendations.head(top_n)

    if recommendations.empty:

        st.warning(
            "No recommendations match the selected filters."
        )

    else:

        st.success(
            f"Showing {len(recommendations)} "
            f"recommendations for {selected_user}"
        )

        st.markdown("### 🏆 Recommended Courses")

        for rank, (_, course) in enumerate(
            recommendations.iterrows(), start=1
        ):

            with st.container(border=True):

                st.markdown(
                    f"#### {rank:02d}. {course['CourseName']}"
                )

                c1, c2 = st.columns(2, gap="medium")

                c1.metric(
                    "📂 Category",
                    course["CourseCategory"]
                )

                c2.metric(
                    "🎓 Level",
                    course["CourseLevel"]
                )

                c3, c4 = st.columns(2, gap="medium")

                if "CourseRating" in course.index:
                    c3.metric(
                        "⭐ Course Rating",
                        f"{course['CourseRating']:.2f}"
                    )

                if "FinalScore" in course.index:
                    c4.metric(
                        "🎯 Recommendation Score",
                        f"{course['FinalScore']:.3f}"
                    )

                if "RecommendationReason" in course.index:
                    st.info(
                        f"💡 {course['RecommendationReason']}"
                    )

        st.divider()

        with st.expander("📋 View Complete Recommendation Table"):

            st.dataframe(
                recommendations,
                hide_index=True,
                width="stretch"
            )



elif page == "Segment Comparison":

    st.markdown("## 📈 Learner Segment Comparison")

    st.caption(
        "Compare learning behavior and engagement "
        "across different learner segments."
    )

    cluster_counts = learner_features["Cluster"].value_counts()
    total_learners = len(learner_features)

    # Segment overview cards
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        with st.container(border=True):

            st.markdown("### 📚 Occasional Learners")

            st.metric(
                "Total Learners",
                f"{cluster_counts.get(0, 0):,}"
            )

            st.metric(
                "Percentage of Learners",
                f"{cluster_counts.get(0, 0) / total_learners * 100:.2f}%"
            )

    with col2:
        with st.container(border=True):

            st.markdown("### 🎯 Highly Engaged Learners")

            st.metric(
                "Total Learners",
                f"{cluster_counts.get(1, 0):,}"
            )

            st.metric(
                "Percentage of Learners",
                f"{cluster_counts.get(1, 0) / total_learners * 100:.2f}%"
            )

    st.divider()

    # Learning behavior metrics
    metrics = [
        "TotalCoursesEnrolled",
        "AverageSpending",
        "AverageCourseRating",
        "DiversityScore",
        "EnrollmentFrequency",
        "LearningDepthIndex"
    ]

    metric_labels = {
        "TotalCoursesEnrolled": "Average Courses Enrolled",
        "AverageSpending": "Average Spending",
        "AverageCourseRating": "Average Course Rating",
        "DiversityScore": "Category Diversity",
        "EnrollmentFrequency": "Enrollment Frequency",
        "LearningDepthIndex": "Learning Depth Index"
    }

    segment_comparison = (
        learner_features.groupby("Cluster")[metrics]
        .mean()
        .round(2)
    )

    segment_comparison.index = segment_comparison.index.map({
        0: "Occasional Learners",
        1: "Highly Engaged, Diverse Learners"
    })

    st.markdown("### 📋 Learning Behavior Comparison")

    st.dataframe(
        segment_comparison.rename(
            columns=metric_labels
        ).T,
        width="stretch"
    )

    st.divider()

    st.markdown("### 📊 Compare Individual Metrics")

    selected_metric = st.selectbox(
        "Select Comparison Metric",
        metrics,
        format_func=lambda x: metric_labels[x]
    )

    col3, col4 = st.columns(2, gap="medium")

    col3.metric(
        "📚 Occasional Learners",
        f"{segment_comparison.loc['Occasional Learners', selected_metric]:.2f}"
    )

    col4.metric(
        "🎯 Highly Engaged Learners",
        f"{segment_comparison.loc['Highly Engaged, Diverse Learners', selected_metric]:.2f}"
    )

    st.bar_chart(
        segment_comparison[selected_metric],
        color="#6366F1"
    )


elif page == "Personalized Learning Path":

    st.markdown("## 🎓 Personalized Learning Path")

    st.caption(
        "Explore your personalized course roadmap "
        "based on your learning preferences."
    )

    selected_user = st.selectbox(
        "👤 Select Learner ID",
        sorted(learner_features["UserID"].unique()),
        key="learning_path_user"
    )

    recommendations = recommend_courses(
        selected_user,
        top_n=5
    )

    if recommendations.empty:

        st.warning("No learning path available.")

    else:

        learning_path = recommendations.copy()

        learning_path.insert(
            0,
            "Step",
            range(1, len(learning_path) + 1)
        )

        st.success(
            f"Personalized Learning Path for {selected_user}"
        )

        col1, col2 = st.columns(2, gap="medium")

        col1.metric(
            "📚 Recommended Courses",
            len(learning_path)
        )

        col2.metric(
            "📂 Learning Categories",
            learning_path["CourseCategory"].nunique()
        )

        st.divider()

        st.markdown("### 🗺️ Your Learning Roadmap")

        for _, course in learning_path.iterrows():

            with st.container(border=True):

                st.markdown(
                    f"#### 🔹 Step {course['Step']}"
                )

                st.markdown(
                    f"### {course['CourseName']}"
                )

                c1, c2 = st.columns(2, gap="medium")

                c1.metric(
                    "📂 Course Category",
                    course["CourseCategory"]
                )

                c2.metric(
                    "🎓 Course Level",
                    course["CourseLevel"]
                )

                c3, c4 = st.columns(2, gap="medium")

                if "CourseRating" in course.index:
                    c3.metric(
                        "⭐ Course Rating",
                        f"{course['CourseRating']:.2f}"
                    )

                if "FinalScore" in course.index:
                    c4.metric(
                        "🎯 Recommendation Score",
                        f"{course['FinalScore']:.3f}"
                    )

                if "RecommendationReason" in course.index:
                    st.info(
                        f"💡 {course['RecommendationReason']}"
                    )

            if course["Step"] < len(learning_path):
                st.markdown(
                    "<div style='text-align:center; "
                    "font-size:30px; color:#6366F1;'>"
                    "↓</div>",
                    unsafe_allow_html=True
                )

        st.success("🏆 End of Your Recommended Learning Path")

        with st.expander("📋 View Complete Learning Path Table"):

            st.dataframe(
                learning_path,
                hide_index=True,
                width="stretch"
            )
