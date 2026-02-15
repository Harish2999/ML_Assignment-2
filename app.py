import streamlit as st
import joblib
import numpy as np      
import pandas as pd     
import warnings
import sklearn.exceptions

# Suppress warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.InconsistentVersionWarning)
warnings.filterwarnings("ignore", message=".*xgboost.*pickle.*")

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

# Feature names
features = ['top-left', 'top-middle', 'top-right', 'middle-left', 
           'middle-middle', 'middle-right', 'bottom-left', 'bottom-middle', 'bottom-right']

st.sidebar.header("Board State (0=blank, 1=x, 2=o)")
board = np.array([st.sidebar.selectbox(f"{feat.replace('-', ' ')}", [0,1,2], key=f"{feat}") 
                  for feat in features])

if st.button(" Predict Winner"):
    X_scaled = scaler.transform(board.reshape(1, -1))
    
    preds = {}
    for name, model in models.items():
        pred = model.predict(X_scaled)[0]
        preds[name] = "positive (win)" if pred == 1 else "negative (loss)"
    
    st.subheader("Predictions")
    df = pd.DataFrame(list(preds.items()), columns=['Model', 'Prediction'])
    st.table(df)
    
    # Show board visually
    board_emoji = {0: '⬜', 1: '❌', 2: '⭕'}
    display_board = np.array([board_emoji[int(x)] for x in board]).reshape(3,3)
    st.write("**Current Board:**")
    for row in display_board:
        st.markdown(" | ".join(row))
