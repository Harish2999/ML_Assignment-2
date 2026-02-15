import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------------
# Streamlit App
# -------------------------------
st.title("ML Assignment 2 - Classification Models")

st.sidebar.header("Upload & Settings")

# Upload dataset
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)

    # If dataset has no headers, allow user to add them
    if st.sidebar.checkbox("Add Tic-Tac-Toe column names"):
        df.columns = [
            "top-left", "top-middle", "top-right",
            "middle-left", "middle-middle", "middle-right",
            "bottom-left", "bottom-middle", "bottom-right",
            "Class"
        ]

    st.write("### Dataset Preview")
    st.write(df.head())

    # Select target column
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)

    # Encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model selection
    model_choice = st.sidebar.selectbox(
        "Choose Model",
        ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    }

    # Train selected model
    model = models[model_choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    if hasattr(model, "predict_proba"):
        if len(np.unique(y)) == 2:
            y_prob = model.predict_proba(X_test)[:,1]
        else:
            y_prob = None
    else:
        y_prob = None

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    # Display metrics
    st.write("### Evaluation Metrics")
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**AUC:** {auc}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**MCC:** {mcc:.4f}")

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Classification Report
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))
else:
    st.info("Please upload a CSV file to proceed.")
