# ML Assignment 2 - Tic-Tac-Toe Win Predictor

**Student**: Harish Kumar K  
**BITS ID**: 2025AB05008  
**Email**: 2025ab05008@wilp.bits-pilani.ac.in  
**Date**: 15-Feb-2026  
**Platform**: BITS Virtual Lab

## a. Problem Statement

Build 6 classification models to predict Tic-Tac-Toe board winning states (positive=win, negative=loss). Deploy interactive Streamlit app on Streamlit Community Cloud with model selection, predictions, and evaluation metrics display.

## b. Dataset Description

**Dataset**: UCI Tic-Tac-Toe Endgame Dataset  
**Source**: https://archive.ics.uci.edu/dataset/98/tic-tac-toe-endgame  
**Size**: 958 instances, 9 features + 1 target  
**Features**: 9 board positions (top-left, top-middle, ..., bottom-right)  
**Encoding**: b(blank)=0, x=1, o=2  
**Target**: positive (win), negative (loss)  
**Train/Test Split**: 80/20 stratified (766/192 samples)

## c. Models Used & Evaluation Metrics

### Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| ------------- | -------- | --- | --------- | ------ | -- | --- |
| Logistic Regression | 0.672 | 0.604 | 0.648 | 0.672 | 0.630 | 0.197 |
| Decision Tree | 0.891 | 0.935 | 0.883 | 0.891 | 0.887 | 0.774 |
| kNN | 0.927 | 0.966 | 0.928 | 0.927 | 0.927 | 0.849 |
| Naive Bayes | 0.911 | 0.947 | 0.914 | 0.911 | 0.912 | 0.819 |
| Random Forest (Ensemble) | 0.953 | 0.984 | 0.955 | 0.953 | 0.953 | 0.905 |
| **XGBoost (Ensemble)** | **0.984** | **0.995** | **0.985** | **0.984** | **0.984** | **0.968** |

**Best Model**: XGBoost (98.4% accuracy) ⭐

### Model Performance Observations

| ML Model Name | Observation about model performance |
| ------------- | ---------------------------------- |
| Logistic Regression | Baseline linear model. Moderate 67% performance due to non-linear Tic-Tac-Toe patterns. |
| Decision Tree | Good 89% accuracy with interpretability. Single tree prone to overfitting. |
| kNN | Excellent 93% using distance-based classification on normalized positions. |
| Naive Bayes | Strong 91% assuming feature independence. Good probabilistic approach. |
| Random Forest (Ensemble) | Bagging achieves 95%. Reduces Decision Tree variance significantly. |
| **XGBoost (Ensemble)** | **Best model (98.4%)**. Gradient boosting captures complex game patterns perfectly. |


## Streamlit App Features
  
✅ Evaluation metrics table  
✅ Classification predictions  

**Live Demo**: [Streamlit App](https://mlassignment-2-mcrfmgsnnnulhw94z3vc2y.streamlit.app/)

## Technologies

ML: scikit-learn 1.6.1, XGBoost 2.0.3
Web: Streamlit
Data: Pandas, NumPy
Deployment: Streamlit Community Cloud

---

**Completed on BITS Virtual Lab**  
**Screenshot in submission PDF**  
**Harish Kumar K | 2025AB05008**

---
