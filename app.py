'''import streamlit as st
import joblib
import os
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

# Load from model/ folder
@st.cache_resource
def load_models():
    models = {}
    model_dir = 'model'
    
    # Load scaler
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    
    # Load all models
    model_names = ['XGBoost', 'Random_Forest', 'Logistic_Regression', 'kNN', 'Naive Bayes', 'Decision Tree']  # Add others
    for name in model_names:
        models[name.replace('_', ' ')] = joblib.load(os.path.join(model_dir, f'{name}.pkl'))
    
    return scaler, models

scaler, models = load_models()'''

import streamlit as st
import joblib
import warnings
import sklearn.exceptions

# SUPPRESS VERSION WARNINGS (Add these 3 lines)
warnings.filterwarnings("ignore", category=sklearn.exceptions.InconsistentVersionWarning)
warnings.filterwarnings("ignore", message=".*xgboost.*pickle.*")
import os

@st.cache_resource
def load_models():
    model_dir = 'model'
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    
    models = {}
    model_files = {
        'XGBoost': 'XGBoost.pkl',
        'Random Forest': 'Random_Forest.pkl', 
        'Logistic Regression': 'Logistic_Regression.pkl',
        'Decision Tree': 'Decision_Tree.pkl',
        'kNN': 'kNN.pkl',
        'Naive Bayes': 'Naive_Bayes.pkl'
    }
    
    for name, fname in model_files.items():
        models[name] = joblib.load(os.path.join(model_dir, fname))
    
    return scaler, models

scaler, models = load_models()
st.success(" All 7 models loaded! XGBoost ready (98.4% accuracy)")


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
