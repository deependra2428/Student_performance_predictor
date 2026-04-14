# Student Performance Prediction using Machine Learning

This project predicts a student's marks, grade, or pass/fail result using a complete machine learning workflow inside a Streamlit app.

## Features

- CSV upload with built-in sample dataset fallback
- Dataset preview and EDA
- Missing value handling and outlier clipping
- Feature selection
- Train-test split
- Model selection for regression and classification
- K-Fold cross-validation
- Evaluation metrics
- Hyperparameter tuning with `GridSearchCV`
- Feature importance view
- Real-time manual prediction form

## Project files

- `app.py` - main Streamlit app
- `data/student_performance_sample.csv` - sample dataset
- `requirements.txt` - Python dependencies

## Run locally

```bash
py -3.11 -m pip install -r requirements.txt
py -3.11 -m streamlit run app.py
```

## Suggested dataset columns

The included sample dataset already contains columns like:

- `study_hours`
- `attendance`
- `previous_marks`
- `sleep_hours`
- `internet_usage_hours`
- `parental_education`
- `family_income_level`
- `extracurricular_participation`
- `final_marks`
- `grade`
- `pass_fail`

