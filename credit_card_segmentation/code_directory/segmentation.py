## Business Goal

Segment credit card customers by income, gender, and credit limit to:
- identify high-value customers for upsell or premium products
- detect under-served segments (low limit but high income)
- tailor marketing and credit line strategies by segment


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 6)

## Load & Inspect Data

df=pd.read_csv("credit.csv")

df.head()

print("Shape:", df.shape)
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nAny nulls by column:")
print(df.isnull().any())


## Basic Cleaning & Encoding

### Strip whitespace from key categorical fields

df['Income_Category'] = df['Income_Category'].str.strip()
df['Gender'] = df['Gender'].str.strip()


### # Ordinal mappings
Gender_map = {'M': 2, 'F': 1}
Income_map = {
    'Less than $40K': 1,
    '$40K - $60K': 2,
    '$60K - $80K': 3,
    '$80K - $120K': 4,
    '$120K +': 5,
    'Unknown': 6
}

df['Gender_ordinal'] = df.Gender.map(Gender_map)
df.head(20)

df['Income_ordinal'] = df['Income_Category'].map(Income_map)
df.head(20)

### Drop rows with missing key fields used for clustering

df = df.dropna(subset=['Income_ordinal', 'Gender_ordinal', 'Credit_Limit'])
print("\nShape after dropping missing key fields:", df.shape)

## Feature Selection & Scaling

clustering_data = df[['Gender_ordinal', 'Income_ordinal', 'Credit_Limit']].copy()

scaler = MinMaxScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

## K-Means Clustering

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(clustering_data_scaled)

df['Credit_card_segments_num'] = clusters
df['Credit_card_segments'] = df['Credit_card_segments_num'].map({
    0: 'Cluster 1',
    1: 'Cluster 2',
    2: 'Cluster 3',
    3: 'Cluster 4',
    4: 'Cluster 5'
})


## Segment Profiles (Business KPIs)

segment_profile = (
    df
    .groupby('Credit_card_segments')
    .agg(
        customers=('Credit_card_segments', 'size'),
        avg_income_band=('Income_ordinal', 'mean'),
        avg_credit_limit=('Credit_Limit', 'mean'),
        avg_gender_code=('Gender_ordinal', 'mean')
    )
    .sort_values('avg_credit_limit', ascending=False)
)
print("\nSegment profile (sorted by avg_credit_limit):")
print(segment_profile)

### Overall reference for credit limit
overall_avg_limit = df["Credit_Limit"].mean()
print("\nOverall average credit limit:", round(overall_avg_limit, 2))

# Define high-value and under-served segments for business story
high_value_segments = segment_profile[
    segment_profile["avg_credit_limit"] > overall_avg_limit
]

under_served_segments = segment_profile[
    (segment_profile["avg_income_band"] >= 4)  # high income
    & (segment_profile["avg_credit_limit"] < overall_avg_limit)
]

print("\nHigh-value segments (above overall avg limit):")
print(high_value_segments)

print("\nUnder-served segments (high income, below avg limit):")
print(under_served_segments)


## Visualisations

df['Credit_card_segments'].value_counts().plot(kind='bar')
plt.title('Customer Count by Credit Card Segment')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.show()


### Average credit limit by segment
segment_profile["avg_credit_limit"].plot(kind="bar")
plt.title("Average Credit Limit by Segment")
plt.xlabel("Segment")
plt.ylabel("Average Credit Limit")
plt.show()

### 3D scatter plot (Gender, Income, Credit Limit)
from mpl_toolkits.mplot3d import Axes3D  # noqa

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for segment in df["Credit_card_segments"].unique():
    seg_data = df[df["Credit_card_segments"] == segment]
    ax.scatter(
        seg_data["Gender_ordinal"],
        seg_data["Income_ordinal"],
        seg_data["Credit_Limit"],
        s=40,
        label=str(segment),
    )

ax.set_xlabel("Gender (2=M, 1=F)", fontsize=11, labelpad=10)
ax.set_ylabel("Income Band (1–5)", fontsize=11, labelpad=10)
ax.set_zlabel("Credit Limit", fontsize=11, labelpad=10)
ax.set_title("3D Scatter Plot of Credit Card Segments", fontsize=14)
ax.legend()
plt.show()

## Business Summary

total_customers = len(df)
max_limit_seg = segment_profile["avg_credit_limit"].idxmax()
min_limit_seg = segment_profile["avg_credit_limit"].idxmin()

print("\n=== Business Summary ===")
print(f"Total customers segmented: {total_customers}")
print(f"Highest-limit segment: {max_limit_seg} "
      f"(avg limit = ${segment_profile.loc[max_limit_seg, 'avg_credit_limit']:.0f})")
print(f"Lowest-limit segment: {min_limit_seg} "
      f"(avg limit = ${segment_profile.loc[min_limit_seg, 'avg_credit_limit']:.0f})")

if not under_served_segments.empty:
    print("\nUnder-served opportunity: high-income segments with below-average limits:")
    print(under_served_segments[["customers", "avg_income_band", "avg_credit_limit"]])
else:
    print("\nNo clear under-served high-income segments detected based on current rules.")

