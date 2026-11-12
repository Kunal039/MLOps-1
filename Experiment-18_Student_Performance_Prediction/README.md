# Experiment 18 – Student Performance Prediction System

**Sample project · CO2/CO3/CO5**

## Aim
To predict a student's final score (regression) and pass/fail risk (classification), compare several algorithms, and give explainable advice.

## Data
`generate_dataset.py` creates 1,000 students with realistic relationships: study hours, attendance, previous score, assignments, sleep, social media, parental education, tutoring and internet access. It also adds data-quality problems: missing values, inconsistent gender labels and duplicates. You can replace `data/student_performance.csv` with real data that uses the same columns.

## Method
- Cleaning: remove duplicates, normalize labels.
- `ColumnTransformer`: median imputation + scaling for numeric columns, one-hot for binary columns, ordinal encoding for education.
- **Regression:** Linear Regression, Decision Tree, Random Forest, SVR.
- **Classification:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN (class-weighted). We report **Fail recall**, because catching at-risk students matters most.
- A fairness check compares accuracy by gender.

## Run
```bash
python generate_dataset.py
python train.py
python predict.py
python predict.py --interactive
```

## Result
**Regression:** Linear Regression is best, with RMSE 5.4 and R² 0.71.

**Classification:**

| Model | Accuracy | F1 | ROC-AUC | Fail recall |
|---|---|---|---|---|
| **SVM** | 0.85 | 0.90 | **0.93** | 0.78 |
| Random Forest | 0.85 | 0.90 | 0.91 | 0.71 |
| KNN | 0.81 | 0.88 | 0.90 | 0.27 (misses at-risk students) |

Accuracy by gender is similar (0.87 / 0.83). Study hours, previous score and attendance are the most important features.
