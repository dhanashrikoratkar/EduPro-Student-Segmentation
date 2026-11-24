"""
EduPro - Step 2: Data preprocessing for learner segmentation.
Run from the Edu_Pro project root: python clustering.py
This step prepares features only; it does not fit K-Means yet.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

INPUT_FILE = Path("learner_features.csv")
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError("learner_features.csv not found. Run feature_engineering.py first.")

df = pd.read_csv(INPUT_FILE)

numeric_features = [
    "TotalCoursesEnrolled", "AverageCourseRating", "AverageSpending",
    "BeginnerEnrollments", "IntermediateEnrollments", "AdvancedEnrollments",
    "DiversityScore", "AverageCoursesPerCategory", "EnrollmentFrequency",
    "LearningDepthIndex", "Age",
]
categorical_features = ["PreferredCategory", "PreferredLevel"]
required = ["UserID", "Gender"] + numeric_features + categorical_features
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"learner_features.csv is missing columns: {missing}")

# Keep UserID as an identifier, not a model feature. Gender is retained in the
# source profile but excluded from behavioral clustering.
work = df[["UserID"] + numeric_features + categorical_features].copy()

# Preserve learners with sparse histories, but don't treat a single enrollment
# as a reliable category/level preference.
sparse_mask = work["TotalCoursesEnrolled"] < 2
work.loc[sparse_mask, categorical_features] = "Unknown"
work["SparseEnrollmentFlag"] = sparse_mask.astype(int)
numeric_features = numeric_features + ["SparseEnrollmentFlag"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

X = preprocessor.fit_transform(work)
feature_names = preprocessor.get_feature_names_out()

processed = pd.DataFrame(X, columns=feature_names)
processed.insert(0, "UserID", work["UserID"].values)

joblib.dump({
    "preprocessor": preprocessor,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "feature_names": list(feature_names),
    "user_ids": work["UserID"].tolist(),
    "X": X,
}, OUTPUT_DIR / "preprocessing_artifacts.pkl")

processed.to_csv(OUTPUT_DIR / "processed_learner_features.csv", index=False)

print("Preprocessing completed successfully.")
print(f"Learners retained: {len(df):,}")
print(f"Numeric features scaled: {len(numeric_features)}")
print(f"Categorical features encoded: {len(categorical_features)}")
print(f"Sparse-history learners flagged: {int(sparse_mask.sum()):,}")
print(f"Processed feature matrix shape: {X.shape}")
print(f"Saved: {OUTPUT_DIR / 'processed_learner_features.csv'}")
print(f"Saved: {OUTPUT_DIR / 'preprocessing_artifacts.pkl'}")
