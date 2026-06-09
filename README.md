# Customer Churn Analysis Using Machine Learning

**Author:** Zaki Abiyu Aqilah   
**Tools:** Python (Pandas, Scikit-learn, Matplotlib, Seaborn)  
**Goal:** Predict customer churn and identify key factors influencing customer departure

---

## Project Overview

This project builds **three machine learning models** (Logistic Regression, Random Forest, and Gradient Boosting) to predict customer churn in a telecommunications company. The models help businesses identify at-risk customers and take proactive retention measures.

**Key Results:**
- **Best Testing Accuracy: 79%** (Logistic Regression & Gradient Boosting)
- Best Precision (Churn): 0.64
- Best Recall (Churn): 0.49
- Most stable model: Logistic Regression

---

## Dataset

The dataset contains **7,043 customer records** with the following variables:

| Variable | Type | Description |
|----------|------|-------------|
| `tenure` | Numerical | Months customer has stayed with company |
| `MonthlyCharges` | Numerical | Monthly bill amount |
| `TotalCharges` | Numerical | Total charges accumulated |
| `gender` | Categorical | Female / Male |
| `SeniorCitizen` | Categorical | Senior citizen status (Yes/No) |
| `Partner` | Categorical | Has partner (Yes/No) |
| `PhoneService` | Categorical | Has phone service (Yes/No) |
| `StreamingTV` | Categorical | Has streaming TV (Yes/No) |
| `InternetService` | Categorical | Type of internet service |
| `PaperlessBilling` | Categorical | Uses paperless billing (Yes/No) |
| `Churn` | Categorical | **Target variable** (Yes/No) |

> **Note:** The original CSV file is not included in this repository. See [`data/README.md`](data/README.md) for data structure details.

---

## Data Split

| Dataset | Samples | Churn Percentage |
|---------|---------|------------------|
| **Training** | 4,865 (70%) | 26.5% |
| **Testing** | 2,085 (30%) | 26.2% |

The dataset maintains consistent churn distribution between training and testing sets.

---

## Model Performance

### Confusion Matrix - Logistic Regression (Best Model)

**Testing Set Results (2,085 cases):**

| Actual \ Predicted | No Churn | Churn |
|:------------------:|:--------:|:-----:|
| **No Churn** | 1,385 | 154 |
| **Churn** | 278 | 268 |

### Performance Metrics Comparison

| Model | Accuracy (Train) | Accuracy (Test) | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
|-------|----------------:|----------------:|------------------:|---------------:|-----------------:|
| **Logistic Regression** | 80% | **79%** | 0.64 | 0.49 | 0.55 |
| **Random Forest** | 100% | 78% | 0.59 | 0.47 | 0.52 |
| **Gradient Boosting** | 82% | **79%** | 0.64 | 0.48 | 0.55 |

### Detailed Classification Report - Logistic Regression

**Testing Set:**

| Class | Precision | Recall | F1-Score | Support |
|-------|----------:|-------:|---------:|--------:|
| No Churn (0) | 0.83 | 0.90 | 0.87 | 1,539 |
| Churn (1) | 0.64 | 0.49 | 0.55 | 546 |

**Macro avg:** 0.74 precision, 0.69 recall, 0.71 f1-score  
**Weighted avg:** 0.78 precision, 0.79 recall, 0.78 f1-score

> **Insight:** The model is better at identifying non-churn customers (90% recall) but struggles with churn customers (only 49% recall). This suggests room for improvement in detecting actual churn cases.

---

## Model Insights

### Why Logistic Regression was chosen as Best Model:

1. **No overfitting** - Stable performance (80% training vs 79% testing)
2. **Simplicity** - Easy to interpret and explain to stakeholders
3. **Computational efficiency** - Fast training and prediction
4. **Competitive performance** - Matches Gradient Boosting with less complexity

### Overfitting Observations:

- **Random Forest** showed severe overfitting (100% training accuracy vs 78% testing)
- Complex models like Random Forest are too flexible for this dataset size

---

## Visualizations Generated

The script produces several visualizations for EDA:

1. **Pie Chart** - Churn distribution overview
2. **Histograms** - Numerical features (tenure, monthly charges, total charges) colored by churn status
3. **Count Plots** - Categorical features distribution across churn classes
4. **Confusion Matrices** - Heatmaps for each model (training & testing)

---

## Example Predictions

Sample predictions on test data (actual vs predicted):

| Tenure | MonthlyCharges | InternetService | PaperlessBilling | Actual Churn | Predicted Churn |
|--------|---------------:|----------------:|-----------------:|-------------:|----------------:|
| 1 | 29.85 | Yes | Yes | No | No |
| 5 | 104.10 | Yes | No | Yes | No |
| 60 | 20.50 | No | Yes | No | No |
| 72 | 115.50 | Yes | Yes | No | No |

---

## Recommendations for Improvement

Based on model limitations (only 49% recall for churn):

1. **Handle class imbalance** - Use SMOTE or class_weight parameter
2. **Feature engineering** - Create interaction terms (e.g., tenure × monthly charges)
3. **Hyperparameter tuning** - GridSearchCV for optimal parameters
4. **Try advanced algorithms** - XGBoost, LightGBM, or Neural Networks
5. **Collect more data** - Especially churn cases to balance classes
6. **Adjust decision threshold** - Lower threshold for churn prediction to increase recall


