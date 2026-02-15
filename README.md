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


ML Assignment 2 - Tic-Tac-Toe Win Predictor
Student: Harish Kumar K
BITS ID: 2025AB05008
Email: 2025ab05008@wilp.bits-pilani.ac.in
Date: 15-Feb-2026
Platform: BITS Virtual Lab

a. Problem Statement
Build 6 classification models to predict Tic-Tac-Toe board winning states (positive=win, negative=loss). Deploy interactive Streamlit app on Streamlit Community Cloud with model selection, predictions, and evaluation metrics display.

b. Dataset Description
Dataset: UCI Tic-Tac-Toe Endgame Dataset
Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/98/tic+ tac+toe+endgame)
Size: 958 instances, 9 features + 1 target
Features: 9 board positions (top-left, top-middle, ..., bottom-right)
Encoding: b(blank)=0, x=1, o=2
Target: positive (win), negative (loss)
Train/Test Split: 80/20 stratified (766/192 samples)

text
Sample Input: xxxbobobob → positive (win)
Board layout:
x x x
b o b
o b o
c. Models Used & Evaluation Metrics
Performance Comparison Table
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.672	0.604	0.648	0.672	0.630	0.197
Decision Tree	0.891	0.935	0.883	0.891	0.887	0.774
kNN	0.927	0.966	0.928	0.927	0.927	0.849
Naive Bayes	0.911	0.947	0.914	0.911	0.912	0.819
Random Forest (Ensemble)	0.953	0.984	0.955	0.953	0.953	0.905
XGBoost (Ensemble)	0.984	0.995	0.985	0.984	0.984	0.968
XGBoost achieved highest performance (98.4% accuracy) ⭐

Model Performance Observations
ML Model Name	Observation about model performance
Logistic Regression	Baseline linear model. Moderate performance (67%) due to non-linear decision boundaries in Tic-Tac-Toe game states.
Decision Tree	Good interpretability with 89% accuracy. Single tree prone to overfitting but handles categorical features well.
kNN	Excellent 93% accuracy using distance-based classification. Works well with normalized board positions.
Naive Bayes	Strong 91% performance assuming feature independence. Effective probabilistic approach for game states.
Random Forest (Ensemble)	Bagging ensemble achieves 95% accuracy. Reduces Decision Tree variance significantly.
XGBoost (Ensemble)	Best model (98.4%). Gradient boosting with regularization handles complex patterns perfectly. Production ready.
Key Insight: Ensemble methods (XGBoost, Random Forest) significantly outperform single models by learning complex game-winning patterns.

Repository Structure
text
ml_assignment-2/
├── app.py                 # Streamlit app with 6 models
├── requirements.txt       # ML dependencies
├── README.md             # This document
└── model/                # Trained models (7 files)
    ├── scaler.pkl
    ├── XGBoost.pkl      # Best model (98.4%)
    ├── Random_Forest.pkl
    ├── Logistic_Regression.pkl
    ├── Decision_Tree.pkl
    ├── kNN.pkl
    └── Naive_Bayes.pkl
Streamlit App Features Implemented
✅ Dataset upload (CSV test data) [1 mark]
✅ Model selection dropdown (6 models) [1 mark]
✅ Evaluation metrics display (table) [1 mark]
✅ Classification predictions (all models) [1 mark]

Live Demo: Streamlit App Link

Development Workflow
text
1. BITS Virtual Lab → model.ipynb (6 models trained)
2. Evaluation metrics computed (Accuracy, AUC, Precision, Recall, F1, MCC)
3. Models saved → model/*.pkl
4. app.py → Interactive Streamlit UI
5. GitHub repo → Deployed on Streamlit Cloud
Technologies Used
text
ML: scikit-learn 1.6.1, XGBoost 2.0.3
Web: Streamlit
Data: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Deployment: Streamlit Community Cloud

Assignment completed on BITS Virtual Lab
Screenshot available in submission PDF
Deadline: 15-Feb-2026 23:59 ✓

Harish Kumar K | 2025AB05008 | SAP MM Consultant | IBM
2025ab05008@wilp.bits-pilani.ac.in

---
