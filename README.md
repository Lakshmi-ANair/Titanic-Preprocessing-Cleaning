# Titanic Data Cleaning and Preprocessing

## Objective
This repository focuses on cleaning and preprocessing the Titanic dataset (`titanic.csv`). The goal is to prepare the raw data into a format suitable for machine learning modeling by handling missing values, encoding categorical features, scaling numerical features, and managing outliers.

## Files Included
*   `Titanic_Data_Cleaning_&_Preprocessing`: Colab notebook containing the Python code for all preprocessing steps.
*   `titanic.csv`: The raw dataset used for this task.

## Steps Performed
The preprocessing workflow involved the following key steps:

1.  **Data Loading and Initial Exploration:**
    *   Loaded the `titanic.csv` dataset using Pandas.
    *   Examined basic information using `.info()`, checked data types, and reviewed the first few rows with `.head()`.
    *   Calculated initial descriptive statistics using `.describe()`.
    *   Identified missing values using `.isnull().sum()`.

2.  **Handling Missing Values:**
    *   **Age:** Missing values were imputed using the **median** age of the available data.
    *   **Embarked:** Missing values (only 2) were imputed using the **mode** (most frequent value) of the 'Embarked' column.
    *   **Cabin:** This column was **dropped** due to a large number of missing values.

3.  **Feature Engineering and Encoding:**
    *   **Dropped Irrelevant Columns:** 'Name', 'Ticket', and 'PassengerId' were removed as they are unlikely to be useful predictive features in their raw form.
    *   **Categorical Encoding:**
        *   'Sex' and 'Embarked' columns were converted into numerical format using **One-Hot Encoding** (via `pd.get_dummies`).
        *   `drop_first=True` was used to avoid multicollinearity (dummy variable trap).

4.  **Feature Scaling:**
    *   Numerical features ('Age', 'Fare', 'SibSp', 'Parch', 'Pclass') were scaled using **Standardization** (Z-score scaling) via Scikit-learn's `StandardScaler`. This transforms the data to have a mean of 0 and a standard deviation of 1.

5.  **Outlier Handling:**
    *   Potential outliers were visualized using **boxplots** for key numerical features like 'Age' and 'Fare'.
    *   Outliers in the **'Fare'** column were identified and removed using the **Interquartile Range (IQR)** method. Data points falling below Q1 - 1.5*IQR or above Q3 + 1.5*IQR were excluded from the final processed dataset.

## Tools Used
*   Python
*   Pandas (for data manipulation)
*   NumPy (for numerical operations)
*   Matplotlib & Seaborn (for visualization, especially boxplots)
*   Scikit-learn (for `StandardScaler`)

## Final Data State
The final processed dataset (`df` or `df_no_outliers` in the notebook) contains only numerical features, has missing values handled, categorical features encoded, relevant numerical features scaled, and outliers removed from the 'Fare' column. This prepared dataset is now more suitable for use in machine learning algorithms.

---

## Possible Questions & Answers

### 1. What are the different types of missing data?

*   **MCAR (Missing Completely At Random):** The probability of data being missing is independent of both the observed and unobserved values. It's the ideal scenario but rare. (e.g., a survey respondent randomly skips a question).
*   **MAR (Missing At Random):** The probability of data being missing depends *only* on the *observed* values of other features, not on the missing value itself. (e.g., men might be less likely to answer a depression survey question - missingness depends on 'gender' which is observed). Imputation methods often assume MAR.
*   **MNAR (Missing Not At Random):** The probability of data being missing depends on the missing value itself or other unobserved factors. This is the hardest case. (e.g., people with very high incomes might be less likely to report their income). Requires careful handling, potentially specialized models.

### 2. How do you handle categorical variables?

*   **Encoding:** Convert them into numerical representations that models can understand.
    *   **One-Hot Encoding:** Creates new binary (0/1) columns for each category. Best for nominal variables (no inherent order, e.g., 'Embarked', 'Sex'). Avoids implying order. Increases dimensionality.
    *   **Label Encoding:** Assigns a unique integer to each category (e.g., 0, 1, 2). Suitable for ordinal variables (inherent order, e.g., 'Low', 'Medium', 'High'). Can mislead models if used on nominal data by implying an order/distance that doesn't exist.
    *   **Dummy Encoding:** Similar to One-Hot, but drops one category level to avoid multicollinearity (as shown with `drop_first=True`).
    *   **Other methods:** Target Encoding, Frequency Encoding, Binary Encoding.
*   **Dropping:** If the variable has too many categories (high cardinality) or seems irrelevant, it might be dropped.
*   **Combining Categories:** Group rare categories into a single 'Other' category.

### 3. What is the difference between normalization and standardization?

*   **Normalization (Min-Max Scaling):** Scales data to a fixed range, usually [0, 1]. Formula: `X_norm = (X - X_min) / (X_max - X_min)`. Sensitive to outliers (they can squash the rest of the data into a small range). Useful when algorithms require data in a specific range (like neural networks, image processing).
*   **Standardization (Z-score Scaling):** Scales data to have a mean of 0 and a standard deviation of 1. Formula: `X_std = (X - mean) / std_dev`. Less sensitive to outliers than normalization. Centers the data. Commonly used for algorithms that assume normally distributed data or are sensitive to feature scales (like SVM, PCA, Logistic Regression).

### 4. How do you detect outliers?

*   **Visualization:**
    *   **Boxplots:** Show data distribution, median, quartiles, and points outside the whiskers (typically 1.5 * IQR) are potential outliers.
    *   **Scatter Plots:** Can reveal points that lie far away from the general cluster of data points, especially in relation to other variables.
    *   **Histograms/Density Plots:** Can show isolated bars or bumps at the extremes of the distribution.
*   **Statistical Methods:**
    *   **Z-score:** Calculate the Z-score for each point (`(X - mean) / std_dev`). Points with a Z-score above a threshold (e.g., > 3 or < -3) are considered outliers. Assumes data is somewhat normally distributed.
    *   **IQR (Interquartile Range) Method:** Identify points falling below Q1 - 1.5*IQR or above Q3 + 1.5*IQR. More robust to non-normally distributed data than Z-score.

### 5. Why is preprocessing important in ML?

*   **"Garbage In, Garbage Out":** ML models learn patterns from data. If the data is noisy, inconsistent, or contains errors, the model will learn incorrect patterns, leading to poor performance and unreliable predictions.
*   **Algorithm Requirements:** Many ML algorithms require input data to be in a specific format (e.g., numerical, no missing values). Some are sensitive to the scale of features.
*   **Improved Model Performance:** Cleaning (handling missing values, outliers) and preprocessing (encoding, scaling) helps algorithms converge faster, learn better patterns, and achieve higher accuracy/better generalization.
*   **Avoiding Bias:** Incorrect handling of missing data or outliers can introduce bias into the model.
*   **Feature Engineering:** Preprocessing is often intertwined with feature engineering, where you create new, more informative features from the raw data.

### 6. What is one-hot encoding vs label encoding?

*   **Label Encoding:** Assigns a unique integer to each category (e.g., Red=0, Green=1, Blue=2).
    *   *Use Case:* Primarily for *ordinal* categorical variables where the order matters (e.g., Small=0, Medium=1, Large=2). Also sometimes used for the *target variable* in classification.
    *   *Problem:* If used on *nominal* data (no order, e.g., countries), it implies an artificial order and distance (e.g., model might think Country 2 is "closer" to Country 1 than Country 10), which can confuse the model.
*   **One-Hot Encoding:** Creates a new binary (0/1) column for each category.
    *   *Use Case:* Primarily for *nominal* categorical variables where order doesn't matter (e.g., 'Sex', 'Embarked').
    *   *Benefit:* Avoids the artificial ordering issue of Label Encoding.
    *   *Drawback:* Can significantly increase the number of features (dimensionality) if the original variable has many categories. Using `drop_first=True` (Dummy Encoding) mitigates perfect multicollinearity.

### 7. How do you handle data imbalance?

Data imbalance occurs when classes in a classification problem are not represented equally (e.g., 95% non-fraud, 5% fraud). This can bias models towards the majority class. Techniques include:

*   **Resampling Techniques** (Applied typically *after* train-test split, only on the *training* data):
    *   **Undersampling:** Randomly remove samples from the *majority* class. Risk: May lose important information.
    *   **Oversampling:** Randomly duplicate samples from the *minority* class. Risk: May lead to overfitting on the minority class.
    *   **SMOTE (Synthetic Minority Over-sampling Technique):** Creates *synthetic* samples of the minority class by interpolating between existing minority samples. Often performs better than simple oversampling.
*   **Algorithmic Approaches:**
    *   **Cost-Sensitive Learning:** Assign different misclassification costs to different classes (penalize misclassifying the minority class more heavily). Many algorithms have a `class_weight` parameter (e.g., Scikit-learn's SVM, Logistic Regression).
    *   **Use Appropriate Evaluation Metrics:** Accuracy is misleading on imbalanced data. Use metrics like Precision, Recall, F1-score, AUC-PR (Area Under Precision-Recall Curve), or AUC-ROC (Area Under Receiver Operating Characteristic Curve).
*   **Generate More Data:** If possible, collect more data, especially for the minority class.

### 8. Can preprocessing affect model accuracy?

**Absolutely, yes, significantly.** Preprocessing is arguably one of the most critical steps affecting model performance.

*   **Positive Effects:** Correct handling of missing values, appropriate encoding, feature scaling suited to the algorithm, and outlier management can drastically improve model accuracy, convergence speed, and generalization ability.
*   **Negative Effects:** Incorrect imputation, using label encoding on nominal data, failing to scale features for sensitive algorithms, or removing outliers inappropriately can severely degrade model performance, lead to biased results, or cause algorithms to fail entirely.
