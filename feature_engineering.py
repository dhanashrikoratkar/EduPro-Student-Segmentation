import pandas as pd
from pathlib import Path

# Expected project structure:
# Edu_Pro/
#   feature_engineering.py
#   data/
#     users.csv
#     courses.csv
#     transactions.csv

DATA_DIR = Path("data")

users = pd.read_csv(DATA_DIR / "users.csv")
courses = pd.read_csv(DATA_DIR / "courses.csv")
transactions = pd.read_csv(DATA_DIR / "transactions.csv")

# Parse dates safely. The supplied dataset uses day/month/year dates.
transactions["TransactionDate"] = pd.to_datetime(
    transactions["TransactionDate"], dayfirst=True, errors="coerce"
)

# Validate essential fields and relationships before creating features.
required = {
    "users": (users, ["UserID", "Age", "Gender"]),
    "courses": (courses, ["CourseID", "CourseCategory", "CourseLevel", "CourseRating"]),
    "transactions": (
        transactions,
        ["TransactionID", "UserID", "CourseID", "TransactionDate", "Amount"],
    ),
}
for table_name, (df, columns) in required.items():
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{table_name} is missing columns: {missing_columns}")
    if df[columns].isna().any().any():
        raise ValueError(f"{table_name} contains missing values in required columns.")

if transactions["TransactionDate"].isna().any():
    raise ValueError("Some TransactionDate values could not be parsed.")
if transactions["Amount"].lt(0).any():
    raise ValueError("Negative transaction amounts found; investigate before proceeding.")
if users["UserID"].duplicated().any() or courses["CourseID"].duplicated().any():
    raise ValueError("Duplicate UserID or CourseID values found.")

# Ensure every transaction can be joined to a known learner and course.
if not transactions["UserID"].isin(users["UserID"]).all():
    raise ValueError("Transactions contain UserIDs absent from users.csv.")
if not transactions["CourseID"].isin(courses["CourseID"]).all():
    raise ValueError("Transactions contain CourseIDs absent from courses.csv.")

# Each row represents one learner-course enrollment in the supplied data.
if transactions.duplicated(["UserID", "CourseID"]).any():
    raise ValueError(
        "Repeated learner-course transactions found. Decide whether these are "
        "re-enrollments or duplicate records before counting courses."
    )

enrollments = transactions.merge(
    courses[
        ["CourseID", "CourseCategory", "CourseLevel", "CourseRating"]
    ],
    on="CourseID",
    how="left",
    validate="many_to_one",
)

# Use the full observed calendar window for a comparable monthly frequency.
observation_months = (
    (transactions["TransactionDate"].max().year - transactions["TransactionDate"].min().year)
    * 12
    + transactions["TransactionDate"].max().month
    - transactions["TransactionDate"].min().month
    + 1
)
observation_months = max(observation_months, 1)

def first_mode(series):
    modes = series.dropna().mode()
    return modes.iloc[0] if not modes.empty else "Unknown"

# Learner-level engagement and preference features.
grouped = enrollments.groupby("UserID")
features = grouped.agg(
    TotalCoursesEnrolled=("CourseID", "nunique"),
    AverageCourseRating=("CourseRating", "mean"),
    AverageSpending=("Amount", "mean"),
    PreferredCategory=("CourseCategory", first_mode),
    PreferredLevel=("CourseLevel", first_mode),
    BeginnerEnrollments=("CourseLevel", lambda s: s.eq("Beginner").sum()),
    IntermediateEnrollments=("CourseLevel", lambda s: s.eq("Intermediate").sum()),
    AdvancedEnrollments=("CourseLevel", lambda s: s.eq("Advanced").sum()),
).reset_index()

# Number of distinct categories explored.
diversity = grouped["CourseCategory"].nunique().rename("DiversityScore")

# Mean number of enrollments per category the learner has explored.
category_counts = (
    enrollments.groupby(["UserID", "CourseCategory"])
    .size()
    .groupby("UserID")
    .mean()
    .rename("AverageCoursesPerCategory")
)

# Average number of enrollments per month in the dataset observation window.
frequency = (
    transactions.groupby("UserID").size().div(observation_months)
    .rename("EnrollmentFrequency")
)

features = (
    features
    .merge(diversity, on="UserID", how="left")
    .merge(category_counts, on="UserID", how="left")
    .merge(frequency, on="UserID", how="left")
)

# LearningDepthIndex is defined here as the share of enrollments at Advanced level.
# It ranges from 0 to 1; it is not a prerequisite or skill-proficiency measure.
level_total = (
    features["BeginnerEnrollments"]
    + features["IntermediateEnrollments"]
    + features["AdvancedEnrollments"]
)
features["LearningDepthIndex"] = (
    features["AdvancedEnrollments"].div(level_total.where(level_total.ne(0))).fillna(0)
)

# Include demographic fields for profile exploration. Do not automatically use
# sensitive/unnecessary demographics as clustering inputs without justification.
features = features.merge(
    users[["UserID", "Age", "Gender"]],
    on="UserID",
    how="left",
    validate="one_to_one",
)

features = features.fillna(
    {
        "AverageCourseRating": 0,
        "AverageSpending": 0,
        "DiversityScore": 0,
        "AverageCoursesPerCategory": 0,
        "EnrollmentFrequency": 0,
        "PreferredCategory": "Unknown",
        "PreferredLevel": "Unknown",
    }
)

features.to_csv("learner_features.csv", index=False)
print(f"Created learner_features.csv with {len(features):,} learner profiles.")
print(f"Observation window: {observation_months} calendar months.")
print(features.head())
