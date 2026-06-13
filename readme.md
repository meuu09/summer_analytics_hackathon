#  E-Commerce Conversion Prediction
### Summer Analytics 2026 — Week 2 Hackathon

---

## Problem Statement

Binary classification task: predict whether an online customer will **convert** (make a purchase) or not, using 12 behavioral and demographic attributes.

- **Target:** `Converted` — 1 (converted) or 0 (not converted)
- **Evaluation Metric:** F1 Score on a hidden private test set
- **Challenge:** Class imbalance — only ~31% of users convert

---

##  Dataset

| File | Rows | Labels | Usage |
|---|---|---|---|
| `train.csv` | 10,000 | Yes | Primary training data |
| `public_test.csv` | 3,000 | Yes | Merged into training data |
| `private_test.csv` | 3,000 | No | Final predictions |

>**Key decision:** Since `public_test.csv` also contains the `Converted` label, it was merged with `train.csv` to create a combined training set of **13,000 labeled rows**.

---

##  Preprocessing

- **Missing Values** — Median imputation (robust to outliers). Imputer fit on training data only to prevent data leakage.
- **Categorical Encoding** — `LabelEncoder` applied to `Device_Type` and `Traffic_Source`.
- **Feature Scaling** — `StandardScaler` (mean=0, std=1) applied after imputation. Required for Logistic Regression convergence and fair coefficient comparison.

> All transformations were fit on training data and applied to private test — no leakage.

---

##  Feature Engineering

12 new features were derived from existing columns to provide richer signals:

| New Feature | Formula | Intuition |
|---|---|---|
| `Engagement_Score` | Pages × Products × (Time + 1) | Overall session intensity |
| `Products_Per_Page` | Products / (Pages + 1) | Focus — how targeted is the user |
| `Time_Per_Page` | Time / (Pages + 1) | Attention depth per page |
| `Is_Returning` | 1 if Previous_Purchases > 0 | Loyal customer — strong conversion signal |
| `Discount_Loyal` | Discount_Seen × Is_Returning | Returning customer who also saw a discount |
| `Income_Per_Age` | Income / (Age + 1) | Purchasing power relative to age |
| `Log_Income` | log(1 + Income) | Normalizes right-skewed income distribution |
| `Log_Time` | log(1 + Time_On_Site) | Normalizes right-skewed session time |
| `Log_Engagement` | log(1 + Engagement_Score) | Smoothed engagement signal |
| `Pages_x_Products` | Pages × Products | Deep browsing interaction term |
| `Discount_x_Pages` | Discount_Seen × Pages_Viewed | Deep browsing with discount seen |
| `Previous_x_Discount` | Previous_Purchases × Discount_Seen | Strong loyalty + discount combo signal |

---

## Model — Logistic Regression

Logistic Regression computes the probability of conversion via a sigmoid function applied to a linear combination of features.

**Parameters:**

| Parameter | Value | Reason |
|---|---|---|
| `class_weight` | `'balanced'` | Auto-weights minority class for 31/69 imbalance |
| `C` | `0.1` | Stronger L2 regularization to reduce overfitting |
| `max_iter` | `1000` | Sufficient iterations for solver convergence |

---

## Validation Strategy

**Stratified 5-Fold Cross-Validation** — ensures each fold maintains the same class distribution as the original dataset.

For each fold:
1. Train model on 4 folds, evaluate on 1 held-out fold
2. Get predicted probabilities for the validation fold
3. Search thresholds from `0.20` to `0.80` in steps of `0.01`
4. Store the threshold that maximizes F1 on the validation fold

**Final threshold** = mean of all 5 fold-optimal thresholds (~0.43), applied to private test predictions.

---

## Results

| Metric | Score |
|---|---|
| Mean CV F1 Score | **~0.565** |
| Baseline (starter notebook) | ~0.45 |

**Improvement attributed to:**
- Merging `public_test.csv` into training data (13K rows)
- 12 engineered features
- Log transforms to reduce skew
- Class balancing with `class_weight='balanced'`
- F1-optimized threshold tuning (not default 0.5)

---

##  Methodology Summary

| Step | Decision | Rationale |
|---|---|---|
| Missing Values | Median Imputation | Robust to outliers |
| Categoricals | Label Encoding | Simple, effective for ML models |
| Feature Engineering | 12 new features | Captures interaction & behavioral signals |
| Scaling | StandardScaler | Required for Logistic Regression |
| Model | Logistic Regression | Principled classifier aligned with course syllabus |
| Imbalance | `class_weight='balanced'` | Prevents majority-class bias |
| Validation | Stratified 5-Fold CV | Preserves class ratio across folds |
| Threshold | F1-tuned per fold | 0.5 is suboptimal for F1 metric |

---

## Repository Structure

```
summer_analytics/
├── data/
│   ├── train.csv
│   ├── public_test.csv
│   ├── private_test.csv
│   └── SA2026_Starter_Notebook.ipynb
├── notebooks/
│   └── notebook.ipynb        # Main submission notebook
├── submission.csv            # Final predictions
├── Report for Summer...pdf   # One-page methodology report
├── requirements.txt
├── .gitignore
└── readme.md

> **Note for reproducibility:** Place all CSV files in the same directory as `notebook.ipynb` before running.

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/meuu09/summer_analytics_hackathon.git
cd summer-analytics_hackathon

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the notebook
jupyter notebook notebook.ipynb
```

---
