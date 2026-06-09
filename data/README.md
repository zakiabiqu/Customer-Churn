# Data Folder

## Dataset Information

Expected file: **`Customer_Churn.csv`**

> **Note:** The CSV file is **not included** in this repository because the data comes from a bootcamp and is for **local / practice purposes only**. Please use a telecommunications customer churn dataset from sources like [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) or the data provided by your bootcamp.

## Data Structure

This dataset contains **7,043 customers** of telecommunication/internet services as of **June 2020**, with 11 features + 1 target.

### Columns Description

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `customerID` | String | Unique identifier for each customer | `7590-VHVEG` |
| `gender` | Categorical | Customer gender | `Female`, `Male` |
| `SeniorCitizen` | Categorical | Whether customer is a senior citizen | `Yes`, `No` |
| `Partner` | Categorical | Whether customer has a partner | `Yes`, `No` |
| `tenure` | Numerical | Number of months customer has stayed with the company | `1` to `72` |
| `PhoneService` | Categorical | Whether customer subscribes to phone service | `Yes`, `No` |
| `StreamingTV` | Categorical | Whether customer subscribes to streaming TV | `Yes`, `No` |
| `InternetService` | Categorical | Type of internet service | `DSL`, `Fiber optic`, `No` |
| `PaperlessBilling` | Categorical | Whether customer uses paperless billing | `Yes`, `No` |
| `MonthlyCharges` | Numerical | Monthly bill amount (USD) | `18.25` to `118.75` |
| `TotalCharges` | Numerical | Total charges accumulated | `0` to `8684.80` |
| `Churn` | Categorical | **Target variable** – whether customer churned | `Yes`, `No` |

### Data Types After Preprocessing (as used in `scripts/model.py`)

| Column | Data Type |
|--------|-----------|
| `gender` | int64 (encoded) |
| `SeniorCitizen` | int64 |
| `Partner` | int64 |
| `tenure` | int64 |
| `PhoneService` | int64 |
| `StreamingTV` | int64 |
| `InternetService` | int64 |
| `PaperlessBilling` | int64 |
| `MonthlyCharges` | float64 |
| `TotalCharges` | float64 |
| `Churn` | int64 (target) |

> `customerID` and `UpdatedAt` (if present) are dropped during preprocessing as they are not relevant for modeling.

## How to Obtain the Data

If you want to run the code, download the dataset from one of these sources:

1. **Kaggle Telco Customer Churn**  
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn

2. **Or use the data from your bootcamp** (must match the column structure above)

After downloading, place the `Customer_Churn.csv` file inside this `data/` folder.

## Sample Row (Before Encoding)

| customerID | gender | SeniorCitizen | Partner | tenure | PhoneService | StreamingTV | InternetService | PaperlessBilling | MonthlyCharges | TotalCharges | Churn |
|------------|--------|--------------|---------|--------|--------------|-------------|-----------------|------------------|----------------|--------------|-------|
| 7590-VHVEG | Female | No | Yes | 1 | No | No | DSL | Yes | 29.85 | 29.85 | No |
| 5575-GNVDE | Male | No | Yes | 60 | Yes | No | Fiber optic | Yes | 20.50 | 1198.80 | No |

## Notes

- The file **will not be uploaded to GitHub** to avoid violating bootcamp policies.
- The script `scripts/model.py` automatically reads `../data/Customer_Churn.csv` – make sure the file exists in the correct location when running the code locally.
- For practice purposes, you can create a synthetic dataset with the same structure.

