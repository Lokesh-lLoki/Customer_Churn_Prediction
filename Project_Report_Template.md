# Customer Churn Prediction — Project Report

## 1. Objective
Build a machine-learning workflow to estimate whether a telecom customer is likely to churn.

## 2. Dataset
- Source: IBM Telco Customer Churn sample dataset
- Rows before cleaning: [fill from notebook]
- Columns before cleaning: [fill from notebook]
- Target: Churn (Yes/No)

## 3. Data preparation
Document actual missing values, duplicate rows, data type corrections, and any records excluded. Explain that `customerID` was excluded as an identifier.

## 4. Exploratory analysis
Insert the churn distribution, tenure, monthly charges, contract type and numeric correlation plots. Write only observations supported by the plots.

## 5. Preprocessing
- Numeric features: median imputation and StandardScaler
- Categorical features: most-frequent imputation and OneHotEncoder
- Train/test split: stratified 80:20

## 6. Model comparison
Insert `model_comparison_cv.csv` results. Explain which model had the highest mean cross-validation F1-score.

## 7. Final test evaluation
- Selected model: [actual output]
- Accuracy: [actual output]
- Precision: [actual output]
- Recall: [actual output]
- F1-score: [actual output]
- ROC-AUC: [actual output]
- Confusion matrix: [insert actual output]

## 8. Conclusion
Summarize the strongest evidence-based churn patterns and actual model performance. Mention limitations and avoid presenting model predictions as certainty.

## 9. Deployment
Describe the Streamlit interface and confirm that the saved preprocessing/model pipeline loads successfully.
