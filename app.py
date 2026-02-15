import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

@st.cache_resource
def load_models():
    model_dir = 'model'
    
    # CHECK if model folder exists
    if not os.path.exists(model_dir):
        st.error("❌ model/ folder missing! Upload model/ folder with 7 PKL files.")
        st.stop()
    
    try:
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
        
        st.success("✅ All 7 models loaded!")
        return scaler, models
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

scaler, models = load_models()

# Features
features = ['top-left', 'top-middle', 'top-right', 'middle-left', 
           'middle-middle', 'middle-right', 'bottom-left', 'bottom-middle', 'bottom-right']

st.title("Tic-Tac-Toe Predictor")

st.sidebar.header("Board (0=blank, 1=x, 2=o)")
board = np.array([
    st.sidebar.selectbox(f"{feat.replace('-', ' ')}", [0,1,2], key=f"p_{i}") 
    for i, feat in enumerate(features)
])

if st.button("Predict Winner"):
    X_scaled = scaler.transform(board.reshape(1, -1))
    
    preds = {}
    for name, model in models.items():
        pred = model.predict(X_scaled)[0]
        preds[name] = "✅ WIN" if pred == 1 else "❌ LOSS"
    
    st.subheader("Model Predictions")
    df = pd.DataFrame(list(preds.items()), columns=['Model', 'Prediction'])
    st.table(df)
    
    # Visual board
    board_emoji = {0: '⬜', 1: '❌', 2: '⭕'}
    display_board = np.array([board_emoji[int(x)] for x in board]).reshape(3,3)
    st.markdown("**Board:**")
    for row in display_board:
        st.markdown(f"**{row[0]} | {row[1]} | {row[2]}**")
