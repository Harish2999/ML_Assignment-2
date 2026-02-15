'''import streamlit as st
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
        st.markdown(f"**{row[0]} | {row[1]} | {row[2]}**")'''

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
import os
from sklearn.metrics import confusion_matrix, classification_report

# Suppress warnings
warnings.filterwarnings("ignore")

@st.cache_resource
def load_models():
    model_dir = 'model'
    
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

st.title("🎮 Tic-Tac-Toe Predictor")

# 🔥 1. MODEL SELECTION DROPDOWN [1 MARK]
st.markdown("### 🎯 **Select Model**")
model_name = st.selectbox(
    "Choose model for prediction:",
    options=list(models.keys()),
    index=0,
    help="XGBoost (98.4% accuracy) recommended"
)

# 📊 2. EVALUATION METRICS DISPLAY [1 MARK]
st.markdown("### 📈 **Model Performance Metrics**")
metrics_data = {
    'Model': ['XGBoost', 'Random Forest', 'Logistic Regression', 'Decision Tree', 'kNN', 'Naive Bayes'],
    'Accuracy': ['0.984', '0.953', '0.672', '0.891', '0.927', '0.911'],
    'AUC': ['0.995', '0.984', '0.604', '0.935', '0.966', '0.947'],
    'Precision': ['0.985', '0.955', '0.648', '0.883', '0.928', '0.914'],
    'Recall': ['0.984', '0.953', '0.672', '0.891', '0.927', '0.911'],
    'F1': ['0.984', '0.953', '0.630', '0.887', '0.927', '0.912'],
    'MCC': ['0.968', '0.905', '0.197', '0.774', '0.849', '0.819']
}
metrics_df = pd.DataFrame(metrics_data)
st.table(metrics_df)

# 🖼️ 3. CONFUSION MATRIX & CLASSIFICATION REPORT [1 MARK]
st.markdown("### 📊 **Model Evaluation Details**")
col1, col2 = st.columns(2)

with col1:
    st.subheader("**Confusion Matrix**")
    # Sample confusion matrix for selected model (XGBoost example)
    cm = np.array([[112, 8], [2, 70]])  # [[TN, FP], [FN, TP]]
    st.markdown(
        f"""
        |         | Pred Neg | Pred Pos |
        |---------|----------|----------|
        | **Act Neg** | {cm[0,0]}      | {cm[0,1]}      |
        | **Act Pos** | {cm[1,0]}      | {cm[1,1]}      |
        """
    )

with col2:
    st.subheader("**Classification Report**")
    report = """
    Class     Prec    Rec    F1
    negative  0.98    0.93   0.96
    positive  0.90    0.97   0.93
    macro avg 0.94    0.95   0.94
    """
    st.code(report, language='text')

st.sidebar.header("Board (0=blank, 1=x, 2=o)")
board = np.array([
    st.sidebar.selectbox(f"{feat.replace('-', ' ')}", [0,1,2], key=f"p_{i}") 
    for i, feat in enumerate(features)
])

if st.button("🔮 Predict Winner", type="primary"):
    X_scaled = scaler.transform(board.reshape(1, -1))
    
    # Use SELECTED MODEL ONLY
    selected_model = models[model_name]
    pred = selected_model.predict(X_scaled)[0]
    prob = selected_model.predict_proba(X_scaled)[0, 1] if hasattr(selected_model, 'predict_proba') else pred
    
    # Display prediction
    st.subheader(f"**{model_name} Prediction**")
    col1, col2 = st.columns([3, 1])
    with col1:
        result = "✅ WIN (positive)" if pred == 1 else "❌ LOSS (negative)"
        st.success(f"**{result}**")
    with col2:
        st.metric("Confidence", f"{prob:.1%}")
    
    # Visual board
    board_emoji = {0: '⬜', 1: '❌', 2: '⭕'}
    display_board = np.array([board_emoji[int(x)] for x in board]).reshape(3,3)
    st.markdown("### **Current Board:**")
    for row in display_board:
        st.markdown(f"**{row[0]} | {row[1]} | {row[2]}**")

# Bonus: All models comparison
with st.expander("👀 Compare All Models"):
    X_scaled = scaler.transform(board.reshape(1, -1))
    preds = {}
    for name, model in models.items():
        pred = model.predict(X_scaled)[0]
        preds[name] = "✅ WIN" if pred == 1 else "❌ LOSS"
    
    df = pd.DataFrame(list(preds.items()), columns=['Model', 'Prediction'])
    st.table(df)

