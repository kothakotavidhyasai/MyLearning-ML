# 🌫️ Air Quality Prediction

> Predicting CO levels from cheap metal oxide sensors using Machine Learning built entirely from scratch with NumPy.

## 📌 Problem Statement
Reference air quality analyzers are expensive lab equipment. This project explores whether cheap metal oxide sensors can accurately predict real CO levels — making air quality monitoring affordable and scalable.

## 📦 Dataset
- **Source**: UCI Air Quality Dataset (Italy, 2004-2005)
- **Size**: 9,357 hourly readings
- **Features**: 13 (after cleaning)
- **Target**: CO(GT) — actual carbon monoxide level

## 🔧 Tech Stack
![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-scratch-orange)
![Pandas](https://img.shields.io/badge/Pandas-EDA-green)
![Sklearn](https://img.shields.io/badge/Sklearn-verify only-red)

## 🗂️ Project Pipeline
```
Raw Data → Cleaning → EDA → PCA → Linear Regression → Evaluation
```

### Step 1 — Data Cleaning
- Fixed European format (separator=`;`, decimal=`,`)
- Discovered hidden missing values stored as `-200` — replaced with `NaN`
- Dropped `NMHC(GT)` — 90% missing
- Applied **KNN Imputation** (k=5) on remaining missing values
- Extracted `hour` and `month` from datetime — rush hour effect confirmed

### Step 2 — EDA Findings
- CO(GT) is right-skewed
- `C6H6(GT)` strongest predictor (correlation = **0.93**)
- Many features 0.8-0.98 correlated → multicollinearity problem
- Hour of day affects CO levels — 8am/6pm rush hours show higher CO
- `PT08.S3(NOx)` inversely correlated with CO (-0.71)

### Step 3 — PCA from Scratch
- Built using covariance matrix + eigenvectors (pure NumPy)
- Verified against sklearn — results matched exactly
- **13 features → 6 components** retaining 95% variance
- PC1 alone captures 48% of all variance

### Step 4 — Linear Regression + Gradient Descent from Scratch
- Implemented MSE cost function
- Batch gradient descent (lr=0.01, 1000 iterations)
- Cost converged around iteration 500
- Zero dependency on sklearn for modeling

## 📊 Results

| Metric | Train | Test |
|--------|-------|------|
| MAE    | 0.29  | 0.30 |
| RMSE   | 0.44  | 0.45 |
| R²     | 0.90  | 0.90 |

✅ **No overfitting** — train and test scores nearly identical
✅ **90% accuracy** — huge improvement over house price project (0.43)
✅ **Mean residual = 0.003** — predictions are unbiased

## 💡 Key Learnings
- Real sensor data hides missing values as `-200` — always inspect value distributions
- KNN imputation is smarter than mean imputation for correlated sensor data
- Higher feature correlation → more aggressive PCA compression (13→6 vs 14→11)
- Linear relationships in features = linear regression works well

## 🚀 How to run
```bash
pip install -r requirements.txt
jupyter notebook Air_Quality_Prediction.ipynb
```

## 🔮 Future Improvements
- Log-transform CO(GT) to handle right skew
- Try Ridge regression to handle remaining multicollinearity
- Add polynomial features to capture non-linear patterns
- Try Random Forest for comparison

## 👤 Author
**Kothakota Vidhyasai** — [GitHub](https://github.com/kothakotavidhyasai)
