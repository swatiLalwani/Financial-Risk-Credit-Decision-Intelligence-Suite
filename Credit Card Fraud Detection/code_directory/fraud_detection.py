# Credit Card Fraud Detection¶

## Executive Summary
On a sample of 14k transactions with 0.14% fraud rate, Isolation Forest catches 15% of fraud with 0.13% false positives. Using a $200–$300 loss assumption per fraud, this prevents an estimated $600–$900 loss in this window. The model serves as a baseline to refine thresholds and features.

## Business Context

This analysis simulates how a card issuer could use anomaly detection to flag potentially fraudulent transactions.

The goal is **not** just to train a model, but to:
- Catch as many fraudulent transactions as possible (high recall on fraud)
- While keeping false alarms at a manageable level for the fraud operations team
- Provide a basis for rules such as "auto-flag" vs "manual review"

## Reading the Data

The dataset is loaded from `CreditcardFraud.csv`, which contains 284,807 transactions with 492 labeled as fraud.This uses the well-known public credit card fraud dataset (European cardholders, September 2013, 284,807 transactions with 492 frauds).


# IMPORTING PACKAGES

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]

##  Load & Understand the Data

df = pd.read_csv('CreditcardFraud.csv',sep=',')
df.head()

## Exploratory Data Analysis

df.info()

df.isnull().values.any()

count_classes = pd.value_counts(df['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")

## Get the Fraud and the normal dataset 

fraud = df[df['Class']==1]

normal = df[df['Class']==0]

print(fraud.shape,normal.shape)

## Analyze more amount of information from the transaction data
#How different are the amount of money used in different transaction classes?
fraud.Amount.describe()

normal.Amount.describe()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();

# How often Do fraudulent transactions occur during certain time frame ? 

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time,fraud.Amount)
ax1.set_title('fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

## Sampling and Handling Imbalance


sdf= df.sample(frac = 0.05,random_state=1)

sdf.shape

df.shape

## Model Training & Evaluation

#Determine the number of fraud and valid transactions in the dataset

Fraud = sdf[sdf['Class']==1]

Valid = sdf[sdf['Class']==0]

outlier_fraction = len(Fraud)/float(len(Valid))

print(outlier_fraction)

print("Fraud Cases : {}".format(len(Fraud)))

print("Valid Cases : {}".format(len(Valid)))

## Correlation
#get correlations of each features in dataset
corrmat = sdf.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(sdf[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#Create independent and Dependent Features
columns = sdf.columns.tolist()
# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)
X = sdf[columns]
Y = sdf[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)

## Model Prediction

##Define the outlier detection methods

classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, 
                                         max_iter=-1)
   
}

type(classifiers)

# Why this matters:
# In fraud analytics, saving customers from false blocks can be more important than squeezing out 2-3% more recall.


n_outliers = len(Fraud)

for clf_name, clf in classifiers.items():
    print("\n==============================")
    print(f"Model: {clf_name}")
    print("==============================")

    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        clf.fit(X)
        y_pred = clf.predict(X)
        scores_prediction = clf.decision_function(X)
    else:
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    y_pred = np.where(y_pred == 1, 0, 1)
    n_errors = (y_pred != Y).sum()

    print("Errors:", n_errors)
    print("Accuracy:", accuracy_score(Y, y_pred))
    print("Classification Report:")
    print(classification_report(Y, y_pred, target_names=LABELS))



# ========================================
# Risk KPIs & Financial Impact for Isolation Forest
# ========================================

from sklearn.metrics import confusion_matrix

print("\n--- Computing Risk KPIs for Isolation Forest ---")

iso_clf = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=outlier_fraction,
    random_state=state,
    verbose=0
)

iso_clf.fit(X)
scores_iso = iso_clf.decision_function(X)
raw_pred = iso_clf.predict(X)

# Convert to 1 = Fraud, 0 = Normal
y_pred_iso = np.where(raw_pred == 1, 0, 1)

# ----------------------------------------
# 1) Core KPIs
# ----------------------------------------
tn, fp, fn, tp = confusion_matrix(Y, y_pred_iso).ravel()

fraud_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
fraud_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
fraud_rate = (tp + fn) / len(Y)
fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\n=== Risk KPIs for Isolation Forest ===")
print(f"Total transactions: {len(Y)}")
print(f"Total fraud cases (true): {tp + fn}")
print(f"Detected fraud cases (TP): {tp}")
print(f"Missed fraud cases (FN): {fn}")
print(f"Fraud recall (TPR): {fraud_recall:.3f}")
print(f"Fraud precision: {fraud_precision:.3f}")
print(f"Fraud rate in sample: {fraud_rate:.5f}")
print(f"False positive rate (FPR): {fp_rate:.5f}")

# ----------------------------------------
# 2) Financial Impact Example
# ----------------------------------------
avg_loss_low = 200
avg_loss_high = 300

total_potential_loss_low = (tp + fn) * avg_loss_low
total_potential_loss_high = (tp + fn) * avg_loss_high

prevented_loss_low = tp * avg_loss_low
prevented_loss_high = tp * avg_loss_high

residual_loss_low = fn * avg_loss_low
residual_loss_high = fn * avg_loss_high

prevented_share_low = prevented_loss_low / total_potential_loss_low if total_potential_loss_low > 0 else 0
prevented_share_high = prevented_loss_high / total_potential_loss_high if total_potential_loss_high > 0 else 0

print("\n=== Example Financial Impact (Assumptions: $200–$300 per fraud) ===")
print(f"Estimated total potential loss: ${total_potential_loss_low:,.0f} – ${total_potential_loss_high:,.0f}")
print(f"Estimated loss prevented by model: ${prevented_loss_low:,.0f} – ${prevented_loss_high:,.0f}")
print(f"Estimated loss still occurring: ${residual_loss_low:,.0f} – ${residual_loss_high:,.0f}")
print(f"Share of loss prevented: {prevented_share_low*100:.1f}% – {prevented_share_high*100:.1f}%")


print("\n### Final KPIs (from current model run)")
print(f"- Fraud rate in sample: {fraud_rate*100:.3f}%")
print(f"- Isolation Forest recall (caught fraud): {fraud_recall*100:.1f}%")
print(f"- False positive rate: {fp_rate*100:.2f}%")
print(f"- Loss prevented estimate: ${prevented_loss_low:,.0f} - ${prevented_loss_high:,.0f}")
print(f"- Share of potential loss prevented: {prevented_share_low*100:.1f}% - {prevented_share_high*100:.1f}%")


## Confusion Matrix Visualization



# Compute confusion matrix from your Isolation Forest predictions
cm = confusion_matrix(Y, y_pred_iso)
labels = ["Normal", "Fraud"]

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Isolation Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


## Anomaly Score Distribution (Fraud vs Normal)

# Plot anomaly score distribution
plt.figure(figsize=(8,5))
sns.histplot(scores_iso, bins=50, kde=True, color="purple")
plt.title("Anomaly Score Distribution - Isolation Forest")
plt.xlabel("Anomaly Score (lower = more anomalous)")
plt.ylabel("Frequency")
plt.show()


## Overlay Fraud Points on Score Plot

fraud_scores = scores_iso[Y == 1]
normal_scores = scores_iso[Y == 0]

plt.figure(figsize=(8,5))
sns.kdeplot(normal_scores, label="Normal", shade=True)
sns.kdeplot(fraud_scores, label="Fraud", shade=True, color="red")
plt.title("Fraud vs Normal - Anomaly Score Density")
plt.xlabel("Anomaly Score")
plt.ylabel("Density")
plt.legend()
plt.show()


## OBSERVATIONS


## Observations & Model Comparison

- **Isolation Forest**
  - Overall accuracy ≈ **99.75%** (expected in a highly imbalanced dataset)
  - Detected ~**15%** of known fraud cases (fraud recall)
  - Very low false-positive rate (~0.13%), so most normal transactions are left untouched.

- **Local Outlier Factor (LOF)**
  - Similar overall accuracy (~**99.71%**), but **0% fraud recall** on this sample
  - Misses all known fraud cases → not usable as a primary fraud model.

- **One-Class SVM**
  - Overall accuracy ≈ **42.4%**
  - High fraud recall (~**70%**) but **near-zero precision** on the fraud class
  - In practice this means it flags a huge number of normal transactions as fraud, creating too much noise for operations.

Because fraud is rare and expensive, **we care most about catching fraud (recall) *and* keeping false positives manageable.**  
On that trade-off, **Isolation Forest is the most practical model** in this experiment: it catches part of the fraud with very few false alarms and provides a good baseline to improve from.



##  Business Interpretation

From a risk perspective, this prototype shows that anomaly detection can:
- Surface a small but meaningful subset of high-risk transactions from a very imbalanced portfolio.
- Provide a fraud score that can drive **tiered actions** (block, review, allow).
- Quantify the trade-off between catching more fraud and creating more false positives.

The current model is a **baseline**: it already prevents part of the potential fraud loss, and can be improved with better features, threshold tuning, and time-based evaluation.


## How This Could Be Used in a Real System

In a production environment, the anomaly score or fraud probability from the best-performing model would be used to drive decisions such as:

- **Score > threshold_high** → automatically block transaction and require additional verification (e.g., OTP, contact center)
- **threshold_low < Score ≤ threshold_high** → allow transaction but flag for fraud team review
- **Score ≤ threshold_low** → treat as normal

These thresholds would be tuned to balance:
- catching more fraud (reducing loss), and  
- limiting friction for genuine customers.

## Limitation:
For this prototype, models were evaluated on the same sample used for fitting (no time-based or hold-out split).  
In a production setting, I would:
- use a time-based split or cross-validation,
- and test performance on a strictly held-out test period to avoid optimistic bias.

## RESULTS

Isolation Forest performed better, gave the best results, and is the most promising model among the three in this experiment.

## NEXT STEPS

--> Tune thresholds

--> Add time-based split validation

--> Add behavioral features (amount deviation, hourly spend, velocity score)

## Model Choice
Given the trade-off between recall and operational noise, I would start with Isolation Forest, then iterate thresholds / features to increase fraud recall while monitoring false positives.

