from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR


st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)

DATA_PATH = Path(__file__).parent / "data" / "student_performance_sample.csv"


class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        frame = pd.DataFrame(X).astype(float)
        q1 = frame.quantile(0.25)
        q3 = frame.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - 1.5 * iqr).to_numpy()
        self.upper_ = (q3 + 1.5 * iqr).to_numpy()
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).astype(float)
        clipped = frame.clip(lower=self.lower_, upper=self.upper_, axis=1)
        return clipped.to_numpy()


REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "SVR": SVR(),
}

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "SVC": SVC(probability=True),
}


REGRESSION_GRIDS = {
    "Linear Regression": {},
    "Random Forest Regressor": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_split": [2, 4],
    },
    "SVR": {
        "model__C": [0.5, 1.0, 5.0],
        "model__kernel": ["rbf", "linear"],
        "model__gamma": ["scale", "auto"],
    },
}

CLASSIFICATION_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.1, 1.0, 10.0],
        "model__solver": ["lbfgs"],
    },
    "Random Forest Classifier": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_split": [2, 4],
    },
    "SVC": {
        "model__C": [0.5, 1.0, 5.0],
        "model__kernel": ["rbf", "linear"],
        "model__gamma": ["scale", "auto"],
    },
}


@st.cache_data
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def build_preprocessor(X: pd.DataFrame, model_name: str) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    needs_scaling = model_name in {"SVR", "SVC", "Logistic Regression"}

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("outliers", OutlierClipper()),
    ]
    if needs_scaling:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("cat", Pipeline(categorical_steps), categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def build_pipeline(X: pd.DataFrame, task_type: str, model_name: str) -> tuple[Pipeline, list[str], list[str]]:
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X, model_name)
    model = REGRESSION_MODELS[model_name] if task_type == "Regression" else CLASSIFICATION_MODELS[model_name]
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline, numeric_cols, categorical_cols


def get_cv(task_type: str, folds: int):
    return KFold(n_splits=folds, shuffle=True, random_state=42) if task_type == "Regression" else StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=42
    )


def run_cross_validation(model, X_train, y_train, task_type: str, folds: int) -> dict[str, float]:
    cv = get_cv(task_type, folds)
    if task_type == "Regression":
        rmse_scores = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        return {
            "cv_rmse_mean": float(rmse_scores.mean()),
            "cv_rmse_std": float(rmse_scores.std()),
            "cv_r2_mean": float(r2_scores.mean()),
        }

    accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
    return {
        "cv_accuracy_mean": float(accuracy_scores.mean()),
        "cv_accuracy_std": float(accuracy_scores.std()),
        "cv_f1_mean": float(f1_scores.mean()),
    }


def evaluate_predictions(task_type: str, y_true, y_pred) -> dict[str, float]:
    if task_type == "Regression":
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        return {
            "RMSE": float(rmse),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
        }

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def feature_rankings(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        importances = np.abs(coef[0]) if np.ndim(coef) > 1 else np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    rankings = pd.DataFrame({"feature": feature_names, "importance": importances})
    return rankings.sort_values("importance", ascending=False).head(10)


def create_manual_input_form(X: pd.DataFrame) -> dict[str, object]:
    entries: dict[str, object] = {}
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    for column in X.columns:
        if column in numeric_cols:
            values = X[column].dropna()
            default = float(values.median()) if not values.empty else 0.0
            minimum = float(values.min()) if not values.empty else 0.0
            maximum = float(values.max()) if not values.empty else max(10.0, default + 1)
            is_integer_like = bool(np.allclose(values, values.round())) if not values.empty else True
            step = 1.0 if is_integer_like else 0.1
            entries[column] = st.number_input(
                column.replace("_", " ").title(),
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=step,
            )
        elif column in categorical_cols:
            options = [str(item) for item in sorted(X[column].dropna().astype(str).unique())]
            entries[column] = st.selectbox(column.replace("_", " ").title(), options=options)
    return entries


def overfitting_message(task_type: str, train_score: float, test_metrics: dict[str, float]) -> str:
    if task_type == "Regression":
        test_score = test_metrics["R2"]
        gap = train_score - test_score
        if gap > 0.15:
            return "Potential overfitting detected: training R2 is noticeably higher than test R2."
        return "No strong overfitting signal detected from the train vs test R2 comparison."

    test_score = test_metrics["Accuracy"]
    gap = train_score - test_score
    if gap > 0.10:
        return "Potential overfitting detected: training accuracy is noticeably higher than test accuracy."
    return "No strong overfitting signal detected from the train vs test accuracy comparison."


st.title("🎓 Student Performance Prediction using Machine Learning")
st.caption(
    "Upload a CSV, explore the data, clean and model it, compare ML algorithms, and make real-time predictions."
)

with st.sidebar:
    st.header("Project Controls")
    task_type = st.selectbox("Select problem type", ["Regression", "Classification"])
    uploaded_file = st.file_uploader("Upload student CSV", type=["csv"])
    use_sample = st.toggle("Use built-in sample dataset", value=uploaded_file is None)
    test_size = st.slider("Test set size (%)", min_value=15, max_value=40, value=20, step=5)
    folds = st.slider("K-Fold splits", min_value=3, max_value=10, value=5, step=1)
    hyperparameter_tuning = st.checkbox("Enable hyperparameter tuning", value=True)

if uploaded_file is not None and not use_sample:
    data = load_uploaded_data(uploaded_file)
    source_label = "uploaded CSV"
else:
    data = load_default_data()
    source_label = "built-in sample dataset"

st.success(f"Loaded {len(data)} rows from the {source_label}.")

st.subheader("1. Dataset Preview")
left_preview, right_preview = st.columns([1.4, 1])
with left_preview:
    st.dataframe(data.head(10), use_container_width=True)
with right_preview:
    st.metric("Rows", data.shape[0])
    st.metric("Columns", data.shape[1])
    st.metric("Missing values", int(data.isna().sum().sum()))

st.subheader("2. Basic EDA")
eda_col1, eda_col2 = st.columns(2)
numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = [col for col in data.columns if col not in numeric_columns]

with eda_col1:
    st.write("Summary statistics")
    st.dataframe(data.describe(include="all").transpose(), use_container_width=True)

with eda_col2:
    if len(numeric_columns) >= 2:
        x_axis = st.selectbox("Scatter plot X-axis", options=numeric_columns, index=0)
        y_default = 1 if len(numeric_columns) > 1 else 0
        y_axis = st.selectbox("Scatter plot Y-axis", options=numeric_columns, index=y_default)
        st.scatter_chart(data[[x_axis, y_axis]], x=x_axis, y=y_axis, use_container_width=True)
    else:
        st.info("Add at least two numeric columns to unlock scatter plot EDA.")

if numeric_columns:
    st.write("Correlation heatmap")
    st.dataframe(data[numeric_columns].corr().round(2), use_container_width=True)

chart_left, chart_right = st.columns(2)
with chart_left:
    if numeric_columns:
        histogram_column = st.selectbox("Histogram column", options=numeric_columns)
        st.bar_chart(
            data[histogram_column]
            .value_counts(bins=min(10, max(4, data[histogram_column].nunique())))
            .sort_index(),
            use_container_width=True,
        )

with chart_right:
    if categorical_columns:
        category_column = st.selectbox("Category distribution", options=categorical_columns)
        st.bar_chart(data[category_column].value_counts().head(10), use_container_width=True)
    else:
        st.info("No categorical columns found for category distribution.")

st.subheader("3. Data Cleaning and Feature Setup")
clean_col1, clean_col2, clean_col3 = st.columns(3)
with clean_col1:
    st.metric("Duplicate rows", int(data.duplicated().sum()))
with clean_col2:
    st.metric("Numeric columns", len(numeric_columns))
with clean_col3:
    st.metric("Categorical columns", len(categorical_columns))

target_options = numeric_columns if task_type == "Regression" else data.columns.tolist()
default_target = "final_marks" if "final_marks" in target_options else target_options[-1]
target_column = st.selectbox("Select target column", options=target_options, index=target_options.index(default_target))

feature_columns = [col for col in data.columns if col != target_column]
selected_features = st.multiselect("Select feature columns", options=feature_columns, default=feature_columns)

if not selected_features:
    st.warning("Select at least one feature column to continue.")
    st.stop()

model_options = list(REGRESSION_MODELS.keys()) if task_type == "Regression" else list(CLASSIFICATION_MODELS.keys())
default_model_name = "Random Forest Regressor" if task_type == "Regression" else "Random Forest Classifier"
model_name = st.selectbox("Select model", options=model_options, index=model_options.index(default_model_name))

X = data[selected_features].copy()
y = data[target_column].copy()

if task_type == "Classification":
    y = y.astype(str)

test_ratio = test_size / 100
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_ratio,
    random_state=42,
    stratify=y if task_type == "Classification" else None,
)

model_pipeline, numeric_used, categorical_used = build_pipeline(X_train, task_type, model_name)

st.info(
    f"Feature selection prepared {len(selected_features)} columns: "
    f"{len(numeric_used)} numeric and {len(categorical_used)} categorical."
)

st.subheader("4. Training, K-Fold Validation, and Tuning")
if st.button("Train Model", type="primary", use_container_width=True):
    with st.spinner("Training model and running validation..."):
        fitted_model = model_pipeline
        best_params = {}

        if hyperparameter_tuning:
            param_grid = REGRESSION_GRIDS[model_name] if task_type == "Regression" else CLASSIFICATION_GRIDS[model_name]
            if param_grid:
                grid = GridSearchCV(
                    estimator=model_pipeline,
                    param_grid=param_grid,
                    cv=get_cv(task_type, folds),
                    scoring="neg_root_mean_squared_error" if task_type == "Regression" else "accuracy",
                    n_jobs=-1,
                )
                grid.fit(X_train, y_train)
                fitted_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                fitted_model.fit(X_train, y_train)
        else:
            fitted_model.fit(X_train, y_train)

        if not best_params and hyperparameter_tuning:
            fitted_model.fit(X_train, y_train)

        cv_scores = run_cross_validation(fitted_model, X_train, y_train, task_type, folds)
        predictions = fitted_model.predict(X_test)
        train_score = fitted_model.score(X_train, y_train)
        test_metrics = evaluate_predictions(task_type, y_test, predictions)
        ranking = feature_rankings(fitted_model)

    st.session_state["trained_model"] = fitted_model
    st.session_state["test_metrics"] = test_metrics
    st.session_state["cv_scores"] = cv_scores
    st.session_state["best_params"] = best_params
    st.session_state["target_column"] = target_column
    st.session_state["selected_features"] = selected_features
    st.session_state["X_frame"] = X
    st.session_state["task_type"] = task_type
    st.session_state["train_score"] = train_score
    st.session_state["feature_rankings"] = ranking
    st.session_state["y_test"] = pd.Series(y_test).reset_index(drop=True)
    st.session_state["predictions"] = pd.Series(predictions).reset_index(drop=True)

if "trained_model" in st.session_state:
    metric_cols = st.columns(3 if task_type == "Regression" else 4)
    for column, (label, value) in zip(metric_cols, st.session_state["test_metrics"].items()):
        column.metric(label, f"{value:.3f}")

    cv_box1, cv_box2, cv_box3 = st.columns(3)
    cv_scores = st.session_state["cv_scores"]
    if task_type == "Regression":
        cv_box1.metric("CV RMSE Mean", f"{cv_scores['cv_rmse_mean']:.3f}")
        cv_box2.metric("CV RMSE Std", f"{cv_scores['cv_rmse_std']:.3f}")
        cv_box3.metric("CV R2 Mean", f"{cv_scores['cv_r2_mean']:.3f}")
    else:
        cv_box1.metric("CV Accuracy Mean", f"{cv_scores['cv_accuracy_mean']:.3f}")
        cv_box2.metric("CV Accuracy Std", f"{cv_scores['cv_accuracy_std']:.3f}")
        cv_box3.metric("CV F1 Mean", f"{cv_scores['cv_f1_mean']:.3f}")

    st.write("Overfitting check")
    st.info(overfitting_message(task_type, st.session_state["train_score"], st.session_state["test_metrics"]))

    if st.session_state["best_params"]:
        st.write("Best hyperparameters")
        st.json(st.session_state["best_params"])

    prediction_compare = pd.DataFrame(
        {
            "Actual": st.session_state["y_test"],
            "Predicted": st.session_state["predictions"],
        }
    ).head(12)
    st.write("Prediction sample")
    st.dataframe(prediction_compare, use_container_width=True)

    ranking = st.session_state["feature_rankings"]
    if not ranking.empty:
        st.write("Top feature importance")
        st.bar_chart(ranking.set_index("feature"), use_container_width=True)

    st.subheader("5. Real-Time Prediction")
    with st.form("prediction_form"):
        manual_inputs = create_manual_input_form(st.session_state["X_frame"][st.session_state["selected_features"]])
        submitted = st.form_submit_button("Predict Student Performance", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([manual_inputs])
        result = st.session_state["trained_model"].predict(input_df)[0]
        if st.session_state["task_type"] == "Regression":
            st.success(f"Predicted {st.session_state['target_column'].replace('_', ' ')}: {float(result):.2f}")
        else:
            st.success(f"Predicted class: {result}")

with st.expander("Viva explanation"):
    st.write(
        """
        This project predicts student performance using machine learning.
        First, the dataset is uploaded and explored through EDA. Then the data is cleaned with missing-value handling,
        outlier clipping, encoding, and feature selection. After that, the system trains models such as Linear Regression,
        Random Forest, SVM, Logistic Regression, and SVC depending on the selected problem type. The pipeline uses
        train-test split, K-Fold validation, evaluation metrics, and optional hyperparameter tuning. Finally, the
        Streamlit app allows real-time prediction from user-entered student details.
        """
    )
