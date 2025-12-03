"""Streamlit app for loan default risk prediction using the LendingClub dataset."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Silence TensorFlow info logs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

DROP_COLUMNS = [
    "desc",
    "mths_since_last_record",
    "annual_inc_joint",
    "dti_joint",
    "verification_status_joint",
    "open_acc_6m",
    "open_il_6m",
    "open_il_12m",
    "open_il_24m",
    "mths_since_rcnt_il",
    "total_bal_il",
    "il_util",
    "open_rv_12m",
    "open_rv_24m",
    "max_bal_bc",
    "all_util",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
    "member_id",
    "emp_title",
    "url",
    "policy_code",
    "application_type",
    "mths_since_last_delinq",
    "zip_code",
    "pymnt_plan",
    "mths_since_last_major_derog",
    "next_pymnt_d",
]

COL_CAT_SELECTED = [
    "term",
    "purpose",
    "grade",
    "sub_grade",
    "verification_status",
    "emp_length",
    "initial_list_status",
    "home_ownership",
    "loan_status",
    "earliest_cr_line",
]

LABEL_ENCODE_COLS = ["term", "sub_grade", "credit_history"]
ONE_HOT_COLS = ["purpose", "verification_status", "emp_length", "home_ownership"]
GRADE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
INITIAL_LIST_STATUS_MAP = {"f": 0, "w": 1}
LOAN_STATUS_MAP = {"bad": 0, "good": 1}
MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = MODEL_DIR / "loan_default_model.keras"
FEATURE_PATH = MODEL_DIR / "feature_columns.json"
HISTORY_PATH = MODEL_DIR / "training_history.json"
DEFAULT_DATA_PATH = Path(__file__).parent / "loan_data_2007_2014.csv"


def load_raw_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    if first_col.lower().startswith("unnamed"):
        df = df.drop(columns=first_col)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["loan_status"] = data["loan_status"].apply(
        lambda x: "good" if x in ("Current", "Fully Paid") else "bad"
    )
    data = data.drop(columns=[col for col in DROP_COLUMNS if col in data.columns])
    data = data.replace({"$": ""}, regex=True)
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = pd.to_numeric(data[col], errors="ignore")
    return data


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in data.columns:
        if data[col].isna().any():
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode().iloc[0])
    return data


def add_credit_history(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["earliest_cr_line"] = pd.to_datetime(
        data["earliest_cr_line"], format="%b-%y", errors="coerce"
    )
    data["earliest_cr_line_year"] = data["earliest_cr_line"].dt.year.clip(upper=2014)
    credit_age = 2014 - data["earliest_cr_line_year"]
    conditions = [credit_age <= 2, credit_age <= 5]
    choices = ["0-2 years", "2-5 years"]
    data["credit_history"] = np.select(conditions, choices, default="+5 years")
    data = data.drop(columns=["earliest_cr_line", "earliest_cr_line_year"])
    data["credit_history"] = data["credit_history"].fillna("+5 years")
    return data


def fit_encoders(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], OneHotEncoder]:
    data = df.copy()
    labelers: Dict[str, LabelEncoder] = {}
    for col in LABEL_ENCODE_COLS:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        labelers[col] = encoder
    data["initial_list_status"] = data["initial_list_status"].map(INITIAL_LIST_STATUS_MAP)
    data["grade"] = data["grade"].map(GRADE_MAP)
    data["loan_status"] = data["loan_status"].map(LOAN_STATUS_MAP)
    one_hot_encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
    one_hot = one_hot_encoder.fit_transform(data[ONE_HOT_COLS])
    one_hot_df = pd.DataFrame(
        one_hot,
        columns=one_hot_encoder.get_feature_names_out(ONE_HOT_COLS),
        index=data.index,
    )
    data = data.drop(columns=ONE_HOT_COLS)
    data = pd.concat([data, one_hot_df], axis=1)
    return data, labelers, one_hot_encoder


def transform_with_encoders(
    df: pd.DataFrame,
    labelers: Dict[str, LabelEncoder],
    one_hot_encoder: OneHotEncoder,
) -> pd.DataFrame:
    data = df.copy()
    for col in LABEL_ENCODE_COLS:
        encoder = labelers[col]
        data[col] = encoder.transform(data[col])
    data["initial_list_status"] = data["initial_list_status"].map(INITIAL_LIST_STATUS_MAP)
    data["grade"] = data["grade"].map(GRADE_MAP)
    data["loan_status"] = data["loan_status"].map(LOAN_STATUS_MAP)
    one_hot = one_hot_encoder.transform(data[ONE_HOT_COLS])
    one_hot_df = pd.DataFrame(
        one_hot,
        columns=one_hot_encoder.get_feature_names_out(ONE_HOT_COLS),
        index=data.index,
    )
    data = data.drop(columns=ONE_HOT_COLS)
    data = pd.concat([data, one_hot_df], axis=1)
    return data


def apply_log_transform(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    data = df.copy()
    for col in numeric_cols:
        if col in data.columns:
            data[col] = np.log1p(data[col].astype(float))
    return data


def prepare_dataset(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    raw_df = load_raw_dataset(path)
    cleaned = basic_clean(raw_df)
    num_cols = cleaned.select_dtypes(include="number").columns
    corr_matrix = cleaned[num_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_tri = corr_matrix.where(mask)
    col_to_drop = [col for col in upper_tri.columns if upper_tri[col].abs().gt(0.9).any()]
    col_num_selected = [col for col in num_cols if col not in col_to_drop]
    desired_cols = [col for col in COL_CAT_SELECTED + col_num_selected if col in cleaned.columns]
    if set(COL_CAT_SELECTED) - set(desired_cols):
        missing = ", ".join(sorted(set(COL_CAT_SELECTED) - set(desired_cols)))
        raise ValueError(f"Missing expected columns: {missing}")
    subset = cleaned[desired_cols].copy()
    subset = impute_missing_values(subset)
    subset = add_credit_history(subset)
    reference_df = subset.copy()
    encoded, labelers, one_hot_encoder = fit_encoders(subset)
    encoded = apply_log_transform(encoded, col_num_selected)
    numeric_stats = reference_df[col_num_selected].agg(["min", "max", "median"]).transpose()
    categorical_options = {
        col: sorted(reference_df[col].dropna().unique().tolist())
        for col in [
            "term",
            "purpose",
            "grade",
            "sub_grade",
            "verification_status",
            "emp_length",
            "initial_list_status",
            "home_ownership",
            "credit_history",
        ]
    }
    meta: Dict[str, Any] = {
        "col_num_selected": col_num_selected,
        "label_encoders": labelers,
        "one_hot_encoder": one_hot_encoder,
        "numeric_stats": numeric_stats,
        "categorical_options": categorical_options,
        "feature_columns": [col for col in encoded.columns if col != "loan_status"],
    }
    return encoded, reference_df, meta


def build_model(input_dim: int) -> Sequential:
    tf.keras.backend.clear_session()
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(input_dim,), kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_or_load_model(
    encoded: pd.DataFrame,
    meta: Dict[str, Any],
    force_retrain: bool = False,
) -> Dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    feature_columns = meta["feature_columns"]
    if FEATURE_PATH.exists():
        stored_features = json.loads(FEATURE_PATH.read_text(encoding="utf-8"))
        if stored_features != feature_columns:
            force_retrain = True
    X = encoded[feature_columns]
    y = encoded["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_smt, y_train_smt = smote.fit_resample(X_train, y_train)
    X_train_smt = X_train_smt.astype("float32")
    y_train_smt = y_train_smt.astype("float32")
    X_test_np = X_test.astype("float32")
    if MODEL_PATH.exists() and not force_retrain:
        model = load_model(MODEL_PATH)
        history_data = json.loads(HISTORY_PATH.read_text(encoding="utf-8")) if HISTORY_PATH.exists() else None
    else:
        model = build_model(X_train_smt.shape[1])
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=0.0001),
        ]
        history = model.fit(
            X_train_smt,
            y_train_smt,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0,
        )
        history_data = {k: [float(val) for val in values] for k, values in history.history.items()}
        model.save(MODEL_PATH, include_optimizer=True)
        FEATURE_PATH.write_text(json.dumps(feature_columns), encoding="utf-8")
        HISTORY_PATH.write_text(json.dumps(history_data), encoding="utf-8")
    test_loss, test_accuracy = model.evaluate(X_test_np, y_test, verbose=0)
    y_prob = model.predict(X_test_np, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    metrics = {
        "loss": float(test_loss),
        "accuracy": float(test_accuracy),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    metrics["fpr"] = fpr
    metrics["tpr"] = tpr
    metrics["thresholds"] = thresholds
    return {
        "model": model,
        "metrics": metrics,
        "history": history_data,
        "feature_columns": feature_columns,
        "X_test": X_test,
        "y_test": y_test,
    }


def transform_user_input(user_input: Dict[str, Any], meta: Dict[str, Any]) -> np.ndarray:
    data = pd.DataFrame([user_input])
    data = transform_with_encoders(
        data,
        meta["label_encoders"],
        meta["one_hot_encoder"],
    )
    data = apply_log_transform(data, meta["col_num_selected"])
    feature_df = data[meta["feature_columns"]]
    return feature_df.astype("float32").to_numpy()


def render_overview(reference_df: pd.DataFrame, meta: Dict[str, Any]) -> None:
    st.subheader("Dataset Snapshot")
    st.dataframe(reference_df.head(20))
    st.caption("Preview after cleaning and feature engineering (before encoding).")
    st.subheader("Class Distribution")
    class_counts = reference_df["loan_status"].value_counts().rename_axis("Loan Status").reset_index(name="Count")
    st.bar_chart(class_counts.set_index("Loan Status"))
    st.subheader("Selected Numeric Features")
    numeric_info = meta["numeric_stats"].copy()
    numeric_info.columns = ["Min", "Max", "Median"]
    st.dataframe(numeric_info)


def render_performance(bundle: Dict[str, Any]) -> None:
    st.subheader("Evaluation Metrics")
    metrics = bundle["metrics"].copy()
    display_metrics = {
        "Loss": metrics["loss"],
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1-Score": metrics["f1"],
        "ROC AUC": metrics["roc_auc"],
    }
    st.dataframe(pd.DataFrame([display_metrics]).transpose(), use_container_width=True)
    st.subheader("Confusion Matrix")
    cm = metrics["confusion_matrix"]
    fig_cm = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"]).figure
    st.pyplot(fig_cm)
    st.subheader("ROC Curve")
    fpr, tpr = metrics["fpr"], metrics["tpr"]
    fig_roc = sns.lineplot(x=fpr, y=tpr).figure
    ax = fig_roc.axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    st.pyplot(fig_roc)
    if bundle["history"]:
        st.subheader("Training Curves")
        history_df = pd.DataFrame(bundle["history"])
        st.line_chart(history_df[[col for col in history_df.columns if "loss" in col]])
        if "accuracy" in history_df.columns and "val_accuracy" in history_df.columns:
            st.line_chart(history_df[["accuracy", "val_accuracy"]])


def render_prediction_form(
    model_bundle: Dict[str, Any],
    meta: Dict[str, Any],
    reference_df: pd.DataFrame,
) -> None:
    st.subheader("Predict Loan Outcome")
    st.write("Fill in the borrower profile to estimate the probability of a good loan.")
    categorical_inputs: Dict[str, Any] = {}
    cat_options = meta["categorical_options"]
    col1, col2, col3 = st.columns(3)
    with col1:
        categorical_inputs["term"] = st.selectbox("Term", cat_options["term"])
        categorical_inputs["grade"] = st.selectbox("Grade", cat_options["grade"])
        categorical_inputs["initial_list_status"] = st.selectbox(
            "Initial List Status", cat_options["initial_list_status"]
        )
    with col2:
        categorical_inputs["purpose"] = st.selectbox("Purpose", cat_options["purpose"])
        categorical_inputs["sub_grade"] = st.selectbox("Sub Grade", cat_options["sub_grade"])
        categorical_inputs["emp_length"] = st.selectbox("Employment Length", cat_options["emp_length"])
    with col3:
        categorical_inputs["verification_status"] = st.selectbox(
            "Verification Status", cat_options["verification_status"]
        )
        categorical_inputs["home_ownership"] = st.selectbox(
            "Home Ownership", cat_options["home_ownership"]
        )
        categorical_inputs["credit_history"] = st.selectbox(
            "Credit History", cat_options["credit_history"]
        )
    st.markdown("---")
    st.subheader("Financial Metrics")
    num_cols = meta["col_num_selected"]
    numeric_stats = meta["numeric_stats"]
    numeric_inputs: Dict[str, float] = {}
    num_columns = st.columns(3)
    for idx, col in enumerate(num_cols):
        widget = num_columns[idx % 3]
        stats_row = numeric_stats.loc[col]
        default = float(stats_row["median"]) if not np.isnan(stats_row["median"]) else 0.0
        numeric_inputs[col] = widget.number_input(
            col,
            value=default,
            format="%.4f",
            help=f"Min: {stats_row['min']:.2f}, Max: {stats_row['max']:.2f}"
        )
    if st.button("Predict"):
        user_payload: Dict[str, Any] = {**categorical_inputs, **numeric_inputs}
        user_payload["loan_status"] = "good"
        features = transform_user_input(user_payload, meta)
        model = model_bundle["model"]
        probability = float(model.predict(features, verbose=0).flatten()[0])
        st.write(f"**Probability of Good Loan:** {probability:.2%}")
        if probability >= 0.5:
            st.success("Model predicts this loan is likely to be GOOD.")
        else:
            st.error("Model predicts this loan is likely to be BAD.")


def main() -> None:
    st.set_page_config(page_title="Loan Default Risk", layout="wide")
    st.title("Loan Default Risk Prediction Dashboard")
    data_path_str = st.sidebar.text_input(
        "Dataset path",
        value=str(DEFAULT_DATA_PATH),
        help="Path to loan_data_2007_2014.csv"
    )
    data_path = Path(data_path_str)
    if not data_path.exists():
        st.error(f"Dataset not found at: {data_path_str}")
        return
    try:
        encoded, reference_df, meta = prepare_dataset(data_path)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Failed to prepare dataset: {exc}")
        return
    if st.session_state.get("data_path") != str(data_path.resolve()):
        st.session_state.pop("model_bundle", None)
        st.session_state["data_path"] = str(data_path.resolve())
    if "model_bundle" not in st.session_state:
        with st.spinner("Training or loading model..."):
            st.session_state["model_bundle"] = train_or_load_model(encoded, meta)
    if st.sidebar.button("Retrain model"):
        with st.spinner("Retraining model..."):
            st.session_state["model_bundle"] = train_or_load_model(encoded, meta, force_retrain=True)
    model_bundle = st.session_state["model_bundle"]
    page = st.sidebar.radio("Navigate", ["Overview", "Model Performance", "Predict"])
    if page == "Overview":
        render_overview(reference_df, meta)
    elif page == "Model Performance":
        render_performance(model_bundle)
    else:
        render_prediction_form(model_bundle, meta, reference_df)


if __name__ == "__main__":
    main()
