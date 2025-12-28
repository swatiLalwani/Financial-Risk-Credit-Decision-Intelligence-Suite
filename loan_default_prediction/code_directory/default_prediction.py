# Executive Summary

The model achieved an AUC of 0.866 and captured 81.52% of high-risk applicants before loan approval. With a $1,500 charge-off assumption per default, the model prevents an estimated $3.96M in losses per approval cycle while leaving $898K in residual exposure. 

By segmenting borrowers into **Auto-Approve / Manual-Review / Decline** tiers, the organization can reduce risk, improve funding decisions, and better control underwriting workload.



## Business Goal
Predict which applicants are at risk of loan default to:
- reduce charge-offs and risk exposure
- prioritize manual review cases
- improve approval strategies


## IMPORTING PACKAGES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
import scikitplot as skplt
# Plot settings
plt.rcParams["figure.figsize"] = (10, 6)
from matplotlib.patches import Ellipse
from sklearn.preprocessing import LabelEncoder
lbl_enc = LabelEncoder()


### READING THE DATASET

data = pd.read_csv('Loan_Repayment.csv')
#getting glimpse of the data
data.head()

## Quick Peak

print("Shape:", data.shape)
print(data.head())

##  BASIC CLEANING & COLUMN NAME NORMALIZATION

#replacing space with '_'
data.columns = data.columns.str.replace(" ", "_")
print("Columns after rename:\n", data.columns)
data.head()

## Checking Missing Values

print("\nMissing values per column:")
print(data.isnull().sum())

## Exploratory Analysis

# Target distribution
sns.countplot(x="Status", data=data)
plt.title("Loan Repayment Status Distribution")
plt.xlabel("Status (1 = Paid, 0 = Default)")
plt.ylabel("Count")
plt.show()


if "CREDIT_Grade" in data.columns:
    credit_order = [
        "A1","A2","A3","A4","A5",
        "B1","B2","B3","B4","B5",
        "C1","C2","C3","C4","C5",
        "D1","D2","D3","D4","D5",
        "E1","E2","E3","E4","E5",
        "F1","F2","F3","F4","F5",
        "G1","G2","G3","G4","G5"
    ]

    plt.figure(figsize=(14, 6))
    sns.countplot(x="CREDIT_Grade", data=data, order=credit_order)
    plt.title("Distribution of Credit Grade")
    plt.xlabel("Credit Grade")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


from matplotlib.patches import Ellipse

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
credit_order = [
    'A1','A2','A3','A4','A5',
    'B1','B2','B3','B4','B5',
    'C1','C2','C3','C4','C5',
    'D1','D2','D3','D4','D5',
    'E1','E2','E3','E4','E5',
    'F1','F2','F3','F4','F5',
    'G1','G2','G3','G4','G5'
]

# Repayment = 1 (paid)
sns.countplot(
    x="CREDIT_Grade",
    data=data[data["Status"] == 1],
    order=credit_order,
    ax=axes[0],
    palette="dark"
)
axes[0].set(
    xlabel="Credit Grade",
    ylabel="Count",
    title="Distribution of Credit Grade: Repayment Status = 1"
)
axes[0].tick_params(axis="x", rotation=45)

# Repayment = 0 (default)
sns.countplot(
    x="CREDIT_Grade",
    data=data[data["Status"] == 0],
    order=credit_order,
    ax=axes[1],
    palette="dark"
)
axes[1].set(
    xlabel="Credit Grade",
    ylabel="Count",
    title="Distribution of Credit Grade: Repayment Status = 0"
)
axes[1].tick_params(axis="x", rotation=45)

# Use axis-relative coordinates for labels instead of huge y-values
axes[0].text(
    0.98, 0.9,
    "Repayment_Status: 1",
    transform=axes[0].transAxes,
    ha="right",
    va="center",
    fontsize=10,
    fontweight="light"
)

axes[1].text(
    0.98, 0.9,
    "Repayment_Status: 0",
    transform=axes[1].transAxes,
    ha="right",
    va="center",
    fontsize=10,
    fontweight="light"
)

# Optional: highlight “most default” region visually but safely
ellipse = Ellipse(
    (0.15, 0.5),   # center in axes fraction (x,y)
    0.18, 0.6,     # width, height (fraction of axes)
    edgecolor="blue",
    facecolor="none",
    linewidth=2,
    transform=axes[1].transAxes
)
axes[1].add_patch(ellipse)

axes[1].annotate(
    "Most default cases",
    xy=(0.15, 0.8),
    xycoords="axes fraction",
    xytext=(0.4, 0.8),
    textcoords="axes fraction",
    arrowprops=dict(connectionstyle="arc3", width=1, color="blue"),
    fontsize=11,
    color="blue",
    fontweight="light"
)

plt.tight_layout()
plt.show()


FICO RANGE

fig,axes = plt.subplots(1,1,figsize=(12,12))
graph = sns.countplot(x='FICO_Range',data = data,ax=axes, palette= 'dark')
graph.set(xlabel='FICO_Range', ylabel='Count')
plt.xticks(rotation=90)
plt.show()

Loan Applications

fig,axes = plt.subplots(1,1,figsize=(10,10))
graph = sns.countplot(x='State',data = data,ax=axes, palette= 'pastel')
graph.set(xlabel='US States', ylabel='Count')
plt.xticks(rotation=90)
plt.show()

Employment Length

fig,axes = plt.subplots(2,1,figsize=(10,10))
categ = ['7 years','1 year','< 1 year','5 years','10+ years','3 years','4 years','6 years','2 years','8 years','n/a','9 years']
sns.countplot(x='Employment_Length', data = data[data['Status'] == 1],ax=axes[0],order=categ, palette= 'deep')
sns.countplot(x='Employment_Length', data = data[data['Status'] == 0],ax=axes[1],order=categ,  palette= 'deep')
plt.show()


data.columns
data.columns = data.columns.str.replace(' ','_')
data.columns

corr = data.corr()
a4_dims = (10, 8)
fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,ax=ax)
plt.show()

data.isnull().sum()

## FEATURE ENGINEERING & ENCODING

if "Interest_Rate" in data.columns:
    data["Interest_Rate"] = data["Interest_Rate"].str.replace("%", "", regex=False)


data.loc[data['Loan_Length'] == '36 months','Loan_Length'] = 36
data.loc[data['Loan_Length'] == '60 months','Loan_Length'] = 60

data['CREDIT_Grade_Code'] = lbl_enc.fit_transform(data['CREDIT_Grade'])

data['Loan_Purpose'] = lbl_enc.fit_transform(data['Loan_Purpose'])

data['DebtToIncomeRatio'] = data['DebtToIncomeRatio'].str.replace('%','')

data['City'] = lbl_enc.fit_transform(data['City'])

data['State'] = lbl_enc.fit_transform(data['State'])

data['Home_Ownership'] = lbl_enc.fit_transform(data['Home_Ownership'])

def comp_avg(row):
    if isinstance(row,str):
        t = row.split('-')
        return (int(t[0]) + int(t[1]))/2
    else:
        return 0
    
data['FICO_Range'] = data['FICO_Range'].apply(comp_avg)
data['FICO_Range'].head(5)

if "FICO_Range" in data.columns:
    data = data.loc[~data["FICO_Range"].isna()]

if "Earliest_CREDIT_Line" in data.columns:
    data["Earliest_CREDIT_Line"] = pd.to_datetime(data["Earliest_CREDIT_Line"], errors="coerce")


data['Revolving_Line_Utilization'] = data['Revolving_Line_Utilization'].str.replace('%','')

data['Employment_Length'] = lbl_enc.fit_transform(data['Employment_Length'])

data['Interest_Rate'] = data['Interest_Rate'].astype(float)
data['Loan_Length'] = data['Loan_Length'].astype(float)
data['DebtToIncomeRatio'] = data['DebtToIncomeRatio'].astype(float)
data['Revolving_Line_Utilization'] = data['Revolving_Line_Utilization'].astype(float)

## TRAIN / TEST SPLIT

data = data.drop(['Months_Since_Last_Record','Months_Since_Last_Delinquency','Education'],axis=1)

data.dropna(how='any',inplace=True)

X = data.drop(['Loan_ID','CREDIT_Grade','Loan_Title','Earliest_CREDIT_Line','Status'],axis=1)
y = data['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

## MODEL – DECISION TREE

dt_model = DecisionTreeClassifier(
    min_samples_split=100,
    random_state=42
)


dt_model.fit(X_train, y_train)

## Predictions

y_pred = dt_model.predict(X_test)
y_proba = dt_model.predict_proba(X_test)[:, 1]


##  METRICS – AUC, CLASSIFICATION REPORT

## 📈 KPI Definitions (Why They Matter)
- **AUC** → How well the model separates good vs. risky borrowers
- **Default Recall (TPR)** → % of risky loans we successfully flag
- **Approval Precision** → % of approved loans that are actually safe

For lending, recall on default is more valuable than accuracy.


auc = roc_auc_score(y_test, y_proba)
print("\nROC-AUC (Decision Tree):", round(auc, 3))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))


## CONFUSION MATRIX

skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
plt.title("Normalized Confusion Matrix – Loan Repayment Model")
plt.show()

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"Loans flagged for review: {fp+tp} ({round((fp+tp)/len(y_test)*100,2)}%)")
print(f"High-risk loans correctly flagged: {tp} (TP)")
print(f"Missed risky loans: {fn} (FN)")


##  ROC CURVE

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"Decision Tree (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Loan Repayment Model")
plt.legend()
plt.show()

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
loss_per_default = 1500

default_recall = tp / (tp + fn)
loss_prevented = tp * loss_per_default
remaining_exposure = fn * loss_per_default


print(f"Default Recall: {default_recall*100:.2f}%")
print(f"Loss prevented: ${loss_prevented:,.0f}")
print(f"Remaining exposure: ${remaining_exposure:,.0f}")




importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=False)

importances.head(10).plot(kind="barh", figsize=(8,6))
plt.title("Top Drivers of Loan Default – Decision Tree")
plt.xlabel("Importance Score")
plt.show()


## 📌 Summary for Stakeholders
- Model AUC: **0.866**
- High-risk drivers: FICO Score ↓, Debt-to-Income ↑, Delinquency history ↑
- Recommended policy action:
  - Auto-approve high FICO + low DTI
  - Manual review medium FICO + high utilization
  - Decline very high DTI + low FICO segments (business rule)


### Final Recommendation
Use this model as a **risk pre-screening tool**, not final approval engine.
Next step: tune threshold for risk tiers (Approve / Review / Decline).


## 🚀 Impact Summary (Non-Technical)
- The model correctly flags **2,642 high-risk applications** before approval.
- This reduces potential default losses while minimizing customer friction.
- We can deploy this as a **pre-screening rule engine** to prioritize manual reviews.


## ===========================
## 📊 Business KPIs for Lending
## ===========================

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Core lending/risk KPIs
default_recall = tp / (tp + fn)                   # How many risky loans we catch
approval_precision = tn / (tn + fp)              # How many approved loans are actually safe
review_rate = (tp + fp) / len(y_test)             # % of loans that require manual review
miss_rate = fn / (tp + fn)                        # % of risky loans we miss

# Business translation
potential_loss_per_default = 1500
estimated_loss_prevented = tp * potential_loss_per_default
estimated_loss_missed = fn * potential_loss_per_default

net_impact = estimated_loss_prevented - estimated_loss_missed

print("======== Business KPIs ========")
print(f"Estimated Loss Prevented:             ${estimated_loss_prevented:,.0f}")
print(f"Estimated Loss Missed (FN):           ${estimated_loss_missed:,.0f}")
print(f"Net Financial Improvement:            ${net_impact:,.0f}")
print("================================\n")
print(f"Default Recall (Risk Capture):        {default_recall*100:.2f}%")
print(f"Safe Approval Precision:              {approval_precision*100:.2f}%")
print(f"Review Rate (Operational Load):       {review_rate*100:.2f}%")
print(f"Miss Rate (Uncaught Risk):            {miss_rate*100:.2f}%")
print("--------------------------------")
print(f"Estimated Loss Prevented:             ${estimated_loss_prevented:,.0f}")
print(f"Estimated Loss Missed (FN):           ${estimated_loss_missed:,.0f}")
print("================================\n")


unnecessary_review_rate = fp / (fp + tp)
print("Unnecessary reviews (false alarms):", round(unnecessary_review_rate*100,2), "%")


kpi_labels = ["Recall (Catch Risk)", "Precision (Safe Approvals)", "Review Load"]
kpi_values = [default_recall, approval_precision, review_rate]

plt.figure(figsize=(8,5))
sns.barplot(x=kpi_labels, y=kpi_values, palette="viridis")
plt.title("📌 Loan Risk Model – Business KPI Overview")
plt.ylim(0,1)
for i, v in enumerate(kpi_values):
    plt.text(i, v + 0.02, f"{v*100:.1f}%", ha="center")
plt.show()


net_impact = loss_prevented - remaining_exposure
print("Net Financial Improvement:", "${:,.0f}".format(net_impact))

# 📌 Business Findings & Interpretation

### 1️⃣ Portfolio Risk Insight
- The model captures **81.52% of risky loans before approval**, reducing exposure to charge-offs.
- Only **21.67% of customers are unnecessarily reviewed**, keeping operations efficient.

### 2️⃣ Drivers of Default Risk
| Driver | Interpretation |
|--------|----------------|
| Low FICO Score | Higher probability of non-payment |
| High Debt-to-Income | Reduced repayment capacity |
| High Utilization | Signs of financial stress |
| Short Employment Tenure | Increased applicant volatility |

### 3️⃣ Operational Recommendation
Segment borrowers using thresholds:

| Tier | Condition | Action |
|------|------------|--------|
| **Low Risk** | High FICO + Low DTI | Auto-Approve |
| **Medium Risk** | Mid-FICO + High Utilization | Manual Review |
| **High Risk** | Low FICO + Very High DTI | Decline / Require Collateral |

### 4️⃣ Business Impact  (Assuming 1500 per missed default)
- 💰 Estimated loss prevented: **$ 3,963,000**
- ⚠️ Remaining exposure: **$ 898,500**
- 🎯 Next step: Increase recall without increasing review load

### 5️⃣ Net Financial Imrovement 
- Net Financial Improvement: $3,064,500


