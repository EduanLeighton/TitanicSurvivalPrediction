# 🚢 Titanic Survival Prediction

## 📌 Project Overview
This project builds and evaluates an end-to-end Machine Learning classification pipeline using Scikit-Learn to predict passenger survival on the Titanic based on demographic and trip-related attributes.

---

## 🔑 Key Components & Workflow

### 1. Environment Setup & Data Loading
* Installs and imports essential Python data science tools (`pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`).
* Loads the built-in Titanic dataset using Seaborn (`sns.load_dataset('titanic')`).

### 2. Feature Selection & Data Cleaning
* **Selected Features:** `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `class`, `who`, `adult_male`, `alone`.
* **Target Variable:** `survived` (binary: 0 = No, 1 = Yes).
* Drops redundant or heavily missing columns like `deck`, `embarked`, and `embark_town`.

### 3. Data Splitting & Stratification
* Accounting for class imbalance (~38% survival rate), the dataset is split into training and testing sets using **Stratified Train-Test Split** (`test_size=0.2`, `random_state=42`).

### 4. Preprocessing Pipeline Construction
* Utilizes `ColumnTransformer` to handle dynamic feature preprocessing:
  * **Numeric Pipeline:** Imputes missing values using the **median** standard strategy, followed by `StandardScaler` feature scaling.
  * **Categorical Pipeline:** Imputes missing values using the **most frequent** category, followed by `OneHotEncoder`.

### 5. Model Optimization & Cross-Validation
* Combines preprocessing with a `RandomForestClassifier` inside an `sklearn.pipeline.Pipeline`.
* Performs hyperparameter optimization using **`GridSearchCV`** across 5 stratified folds (`StratifiedKFold`):
  * Explores `n_estimators`, `max_depth`, and `min_samples_split` parameters.
