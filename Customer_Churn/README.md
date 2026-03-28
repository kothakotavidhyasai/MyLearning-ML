
# Customer Churn Prediction

Number of Columns:
- 21

Target Column:
- `Churn`

---

## Features Used

Some important features in the dataset:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Contract Type
- Monthly Charges
- Total Charges

Target:

- `Churn = 1` → Customer leaves
- `Churn = 0` → Customer stays

---

## Project Workflow

1. Load the dataset
2. Remove unnecessary columns (`customerID`)
3. Convert `TotalCharges` into numeric format
4. Handle missing values
5. Encode target column (`Yes/No → 1/0`)
6. Separate features and labels
7. Split dataset into training and testing sets
8. Perform one-hot encoding on categorical variables
9. Scale numerical features
10. Train Logistic Regression model
11. Evaluate model using multiple metrics
12. Plot ROC Curve

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
