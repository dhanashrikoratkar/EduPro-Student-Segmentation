
import pandas as pd

df = pd.read_csv("Data/transactions.csv")

df["TransactionDate"] = pd.to_datetime(
    df["TransactionDate"],
    format="%d/%m/%Y"
)

first_half = df[
    df["TransactionDate"].dt.month <= 6
]

second_half = df[
    df["TransactionDate"].dt.month >= 7
]

h1 = len(first_half)
h2 = len(second_half)

change = ((h2 - h1) / h1) * 100

results = pd.DataFrame({
    "Metric": [
        "January-June Enrollments",
        "July-December Enrollments",
        "Enrollment Change (%)"
    ],
    "Value": [
        h1,
        h2,
        round(change, 2)
    ]
})

print("\nHISTORICAL ENGAGEMENT PROXY")
print(results.to_string(index=False))

results.to_csv(
    "artifacts/engagement_proxy.csv",
    index=False
)

print("\nEngagement proxy saved successfully!")
