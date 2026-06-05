# ⚡ EV Vehicle Charge Demand Forecasting

> **Predicting Electric Vehicle adoption and charge demand trends using machine learning on real-world county-level data.**

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?style=flat-square&logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-purple?style=flat-square&logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Internship](https://img.shields.io/badge/AICTE-Virtual%20Internship-red?style=flat-square)

---

## 📌 Overview

This project builds an end-to-end **machine learning pipeline to forecast Electric Vehicle (EV) adoption and charge demand** at the county level using real-world population data.

The pipeline covers the full data science workflow — raw data ingestion, cleaning, outlier handling, feature engineering, model training with hyperparameter tuning, and evaluation — producing a production-ready Random Forest model that forecasts EV population trends across geographic regions.

> Built as part of the **AICTE Data Analytics Virtual Internship** under the Green Skills initiative by Shell India Markets Pvt. Ltd. & Edunet Foundation.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | Electric Vehicle Population Data (County-level, USA) |
| **Records** | 20,819 rows |
| **Features** | 10 columns |
| **Target** | EV population count / charge demand per county |
| **Type** | Real-world government dataset |

### Features Used
- County, State (geographic identifiers)
- EV Type (BEV / PHEV)
- Make, Model, Model Year
- Electric Range
- Base MSRP
- Legislative District
- Vehicle Location

---

## 🔬 Methodology

### Full Pipeline

```
Raw Dataset (20,819 rows)
        ↓
Data Cleaning
(null handling, type conversion, duplicates)
        ↓
Outlier Detection & Capping
(IQR-based — Electric Range, Base MSRP)
        ↓
Feature Engineering
(Lag features, Rolling Mean, % Change)
        ↓
Label Encoding
(Categorical → Numerical)
        ↓
Train-Test Split
        ↓
Random Forest Regressor
+ RandomizedSearchCV (Hyperparameter Tuning)
        ↓
Model Evaluation
(MAE, MSE, R²)
        ↓
Forecasting Output
```

---

## ⚙️ Technical Details

### 1. Data Cleaning
- Handled missing values via column-specific imputation strategies
- Removed duplicate records
- Corrected data types for numerical and categorical columns
- Standardized county and state name formats

### 2. Outlier Handling — IQR Method
Applied **Interquartile Range (IQR) capping** to skewed numerical features:
```
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```
Values outside bounds are capped (not removed) to preserve data volume while reducing skew — critical for a 20K+ row real-world dataset with natural outliers.

### 3. Feature Engineering
Three temporal/trend features engineered to capture adoption momentum:

| Feature | Description |
|---|---|
| **Lag Features** | Previous period EV count per county — captures serial dependency |
| **Rolling Mean** | Moving average over N periods — smooths short-term noise |
| **Percentage Change** | Period-over-period growth rate — captures adoption velocity |

### 4. Encoding
`LabelEncoder` applied to categorical columns (County, State, EV Type, Make, Model) to convert to numerical representations compatible with the Random Forest model.

### 5. Model — Random Forest Regressor
Selected for:
- Robustness to outliers
- Handles non-linear relationships
- No feature scaling required
- Built-in feature importance ranking

### 6. Hyperparameter Tuning — RandomizedSearchCV
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}
```
`RandomizedSearchCV` with cross-validation used to efficiently search the hyperparameter space and select the optimal configuration.

### 7. Evaluation Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average prediction error magnitude |
| **MSE** | Mean Squared Error — penalizes large errors |
| **R²** | Coefficient of Determination — variance explained by the model |

---

## 📁 Repository Structure

```
EV_Vehicle_Charge_Demand/
│
├── EV_Adoption_Forecasting.ipynb   # Full pipeline notebook
├── requirements.txt                 # Dependencies (if applicable)
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/AryaBadugu/EV_Vehicle_Charge_Demand.git
cd EV_Vehicle_Charge_Demand
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 3. Run the notebook
```bash
jupyter notebook EV_Adoption_Forecasting.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

---

## 🔑 Key Insights

- **EV adoption is geographically concentrated** — a small number of counties account for a disproportionate share of EV registrations
- **BEV adoption outpaces PHEV** in counties with stronger infrastructure
- **Electric Range and Model Year** are the strongest predictors of adoption in the feature importance ranking
- **Lag features significantly improve forecast accuracy** — past adoption is the best predictor of near-future adoption
- **IQR capping** on Base MSRP and Electric Range reduced model error more than any other preprocessing step

---

## 🏆 Project Context

This project was completed as part of the:

**AICTE Data Analytics Virtual Internship**
*Green Skills Initiative*
Organized by: **AICTE**, **Shell India Markets Pvt. Ltd.**, and **Edunet Foundation**

The internship focused on applying data analytics to sustainability and green technology domains, with EV adoption forecasting as a capstone deliverable.

---

## 👤 Author

**Arya Badugu**
B.E. — Artificial Intelligence & Data Science
SIES Graduate School of Technology, Nerul, Navi Mumbai (2024–2028)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-arya--badugu-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/arya-badugu)
[![GitHub](https://img.shields.io/badge/GitHub-AryaBadugu-black?style=flat-square&logo=github)](https://github.com/AryaBadugu)

---

## 📜 License

This project is licensed under the MIT License.

---

<p align="center">
  Built with ⚡ by <a href="https://github.com/AryaBadugu">Arya Badugu</a> | SIES GST, Navi Mumbai
</p>
