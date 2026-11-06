# Customer Churn Prediction — End-to-End ML Project

## Objective
Predict whether a telecom customer is likely to leave the service using the IBM Telco Customer Churn sample dataset.

## Files
- `Customer_Churn_Prediction.ipynb` — complete notebook: data checks, EDA, IQR outlier review, preprocessing, cross-validation, model selection, test evaluation and saving.
- `app.py` — Streamlit prediction interface.
- `requirements.txt` — Python dependencies.
- `Telco-Customer-Churn.csv` — optional local dataset; the notebook downloads it from IBM if absent.

## Setup (Windows)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the notebook
Open `Customer_Churn_Prediction.ipynb` in Jupyter and run all cells from top to bottom. It will create:
- `churn_model.joblib`
- `model_comparison_cv.csv`
- `test_metrics.json`

The notebook chooses a model using cross-validation on training data and evaluates the selected pipeline once on the held-out test set.

## Run the Streamlit app
After the notebook has successfully created `churn_model.joblib`:
```bash
streamlit run app.py
```

## Dataset source
IBM Telco Customer Churn sample dataset:
https://github.com/IBM/telco-customer-churn-on-icp4d

## Important
Do not claim accuracy, F1-score, row counts after cleaning, or a best-performing model until you run the notebook and record the actual output. This is an educational sample dataset, not live customer data. Predictions are estimates, not guarantees.
