import streamlit as st
import joblib
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

st.title("Tic-Tac-Toe Win Predictor")

# Load scaler and models
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('scaler.pkl')
    models = {}
    model_names = ['Logistic Regression', 'Decision Tree', 'kNN', 'Naive Bayes', 'Random Forest', 'XGBoost']
    for name in model_names:
        models[name] = joblib.load(f'{name.replace(" ", "_")}.pkl')
    return scaler, models

scaler, models = load_artifacts()

# Feature names from dataset
features = ['top-left', 'top-middle', 'top-right', 'middle-left', 'middle-middle', 
            'middle-right', 'bottom-left', 'bottom-middle', 'bottom-right']

# Inputs (encode as 0=b/blank, 1=x, 2=o manually)
st.sidebar.header("Board State (0=blank, 1=x, 2=o)")
board = np.array([st.sidebar.selectbox(f"{feat.replace('-', ' ')}", [0,1,2]) for feat in features])

if st.button("Predict Winner"):
    X_scaled = scaler.transform(board.reshape(1, -1))
    preds = {}
    for name, model in models.items():
        pred = model.predict(X_scaled)[0]
        preds[name] = "positive (win)" if pred == 1 else "negative (loss)"
    
    st.subheader("Predictions")
    df = pd.DataFrame(list(preds.items()), columns=['Model', 'Prediction'])
    st.table(df)
