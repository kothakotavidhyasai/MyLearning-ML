# 🫀 Heart Disease Prediction

> Can Machine Learning save lives? This project builds and compares three classification models to predict heart disease from clinical data — with one model built entirely from scratch.

## 📌 Problem Statement
Heart disease is the leading cause of death worldwide. Early detection is critical. This project explores whether clinical measurements like blood pressure, cholesterol, and chest pain type can accurately predict heart disease — making early screening faster and more accessible.

## 📦 Dataset
- **Source**: Cleveland Heart Disease Dataset (UCI / Kaggle)
- **Size**: 297 patients
- **Features**: 13 clinical measurements
- **Target**: condition (0 = No Disease, 1 = Disease)
- **Class balance**: 54% no disease, 46% disease — well balanced!

## 🔧 Tech Stack
- **Python** — core language
- **NumPy** — logistic regression built from scratch
- **Pandas** — data manipulation
- **Scikit-learn** — SVM, Decision Tree, metrics
- **Matplotlib/Seaborn** — visualizations

## 🗂️ Project Pipeline
```
Raw Data → Cleaning → EDA → Logistic Regression → SVM → Decision Tree → Comparison
```

## 🔍 Step by Step

### Step 1 — Data Cleaning
- Zero missing values — cleanest dataset in this series!
- Capped outliers using **Winsorizing** (5th-95th percentile)
- Scaled continuous features: age, BP, cholesterol, heart rate, oldpeak
- Stratified train/test split — preserved class balance

### Step 2 — EDA Findings
- `thal` strongest predictor (correlation = 0.52)
- `thalach` (max heart rate) inversely correlated — healthy patients reach higher heart rates!
- `cp=3` (chest pain type 3) overwhelmingly associated with disease
- `ca` and `exang` also strong predictors

### Step 3 — Logistic Regression from Scratch
- Implemented sigmoid function from scratch
- Binary cross entropy loss function
- Gradient descent (lr=0.5, 1000 iterations)
- Converged around iteration 700

### Step 4 — SVM & Decision Tree
- SVM with RBF kernel
- Decision Tree with max_depth=5
- Both trained using scikit-learn

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| **Logistic Regression** | **91.67%** | **1.000** | 0.821 | **0.902** | **0.954** |
| SVM | 91.67% | 1.000 | 0.821 | 0.902 | 0.950 |
| Decision Tree | 81.67% | 0.815 | 0.786 | 0.800 | 0.827 |

## 🏆 Best Model — Logistic Regression

- Highest AUC (0.954) — best at ranking patients by disease risk
- Perfect Precision — zero false alarms
- Most interpretable — doctors can understand feature contributions
- **Recommended as a first-line screening tool**

## ⚠️ Limitations
- Recall of 0.82 means 18% of sick patients are missed
- Should be used as screening tool, not final diagnosis
- Dataset is small (297 patients) — needs validation on larger data

## 💡 Key Learnings
- ROC-AUC is more informative than accuracy alone
- In medical diagnosis **Recall matters more than Precision**
- Logistic Regression can match SVM on small clean datasets
- Feature scaling is critical for SVM and Logistic Regression

## 🔮 Future Improvements
- Try Random Forest and XGBoost for higher recall
- Use cross-validation for more reliable evaluation
- Add SHAP values for better model explainability
- Collect more patient data to improve generalization

## 👤 Author
**Kothakota Vidhyasai** — [GitHub](https://github.com/kothakotavidhyasai)

