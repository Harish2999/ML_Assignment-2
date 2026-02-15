# ML_Assignment-2
ML Assignment 2 - Classification Models

## 1. Problem Statement
This project implements multiple machine learning classification models on a chosen dataset to compare their performance using standard evaluation metrics. The models are integrated into a Streamlit web application for interactive demonstration and deployed on Streamlit Community Cloud.

---

## 2. Dataset Description
- **Dataset Source:** [UCI Tic-Tac-Toe Endgame Dataset](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame)  
- **Instances:** 958  
- **Features:** 9 categorical features (board positions)  
- **Target:** `Class` (positive / negative outcome)  
- **Preprocessing:** Label encoding applied to categorical features.

---

## 3. Models Implemented
The following models were trained and evaluated on the dataset:
1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbor Classifier  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

## 4. Evaluation Metrics
The models were evaluated using:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

### Comparison Table

| ML Model Name        | Accuracy | AUC | Precision | Recall | F1 | MCC |
|----------------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression  |          |     |           |        |    |     |
| Decision Tree        |          |     |           |        |    |     |
| kNN                  |          |     |           |        |    |     |
| Naive Bayes          |          |     |           |        |    |     |
| Random Forest        |          |     |           |        |    |     |
| XGBoost              |          |     |           |        |    |     |

*(Fill in values from your notebook results)*

---

## 5. Observations

| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  |                                      |
| Decision Tree        |                                      |
| kNN                  |                                      |
| Naive Bayes          |                                      |
| Random Forest        |                                      |
| XGBoost              |                                      |

*(Add qualitative notes, e.g., “Random Forest performed best overall with high accuracy and balanced precision/recall. Naive Bayes was fastest but less accurate.”)*

---

## 6. Streamlit App Features
- Dataset upload option (CSV) 
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix visualization  

---

## 7. Deployment
- **GitHub Repository:** [Insert Link]  
- **Live Streamlit App:** [Insert Link]  
- **Screenshot:** [Insert Screenshot from BITS Virtual Lab]  

---

## 8. Requirements
Dependencies listed in `requirements.txt` :
numpy
pandas
matplotlib
seaborn
ucimlrepo
sklearn
xgboost

---
