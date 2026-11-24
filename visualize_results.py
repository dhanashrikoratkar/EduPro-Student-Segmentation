
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"

# Load final model comparison
df = pd.read_csv(
    ARTIFACTS / "final_model_comparison.csv"
)

# Convert metrics to percentages
df["Precision@5"] *= 100
df["HitRate@5"] *= 100

# Create chart
ax = df.plot(
    x="Method",
    y=["Precision@5", "HitRate@5"],
    kind="bar",
    figsize=(12, 6),
    width=0.75
)

plt.title(
    "EduPro - Recommendation Model Performance",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Recommendation Method")
plt.ylabel("Performance (%)")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Evaluation Metric")

plt.tight_layout()

# Save chart
plt.savefig(
    ARTIFACTS / "recommendation_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nModel comparison chart saved successfully!")

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"

# Load final model comparison
df = pd.read_csv(
    ARTIFACTS / "final_model_comparison.csv"
)

# Convert metrics to percentages
df["Precision@5"] *= 100
df["HitRate@5"] *= 100

# Create chart
ax = df.plot(
    x="Method",
    y=["Precision@5", "HitRate@5"],
    kind="bar",
    figsize=(12, 6),
    width=0.75
)

plt.title(
    "EduPro - Recommendation Model Performance",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Recommendation Method")
plt.ylabel("Performance (%)")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Evaluation Metric")

plt.tight_layout()

# Save chart
plt.savefig(
    ARTIFACTS / "recommendation_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nModel comparison chart saved successfully!")

# STEP 14.2: Learner Cluster Distribution

clustered = pd.read_csv(
    ARTIFACTS / "clustered_learners.csv"
)

cluster_counts = (
    clustered["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_names = [
    "Occasional Learners",
    "Highly Engaged Learners"
]

plt.figure(figsize=(8, 6))

plt.pie(
    cluster_counts.values,
    labels=cluster_names,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#3182ce", "#f59e0b"]
)

plt.title(
    "EduPro - Learner Cluster Distribution",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    ARTIFACTS / "learner_cluster_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nCluster distribution chart saved successfully!")

# STEP 14.3: Learner Behavior Comparison

profiles = pd.read_csv(
    ARTIFACTS / "cluster_profiles.csv"
)

metrics = [
    "TotalCoursesEnrolled",
    "EnrollmentFrequency",
    "DiversityScore"
]

labels = [
    "Courses Enrolled",
    "Enrollment Frequency",
    "Course Diversity"
]

# Normalize metrics for comparison
comparison = profiles.set_index("Cluster")[metrics].T

comparison = comparison.div(
    comparison.max(axis=1),
    axis=0
)

comparison.index = labels

ax = comparison.plot(
    kind="bar",
    figsize=(10, 6),
    color=["#3182ce", "#f59e0b"]
)

plt.title(
    "EduPro - Learner Behavior by Cluster",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Behavior Metric")
plt.ylabel("Relative Value (0 to 1)")
plt.xticks(rotation=0)
plt.legend([
    "Occasional Learners",
    "Highly Engaged Learners"
])

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    ARTIFACTS / "learner_behavior_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nLearner behavior comparison chart saved successfully!")

