# Experiment 2 – Linear Regression for Prediction

**Unit 1 · CO3** | Syllabus: *Implement linear regression with Scikit-learn, train on a sample dataset (house price prediction), evaluate using MSE.*

## Aim
To build a linear regression model that predicts house prices and to evaluate it with MSE, RMSE, MAE and R².

## Theory
Linear regression fits `y = β0 + β1x1 + … + βnxn` by minimizing the **Mean Squared Error** `MSE = (1/n)Σ(y − ŷ)²`.
- **RMSE** = √MSE, in the same units as the target.
- **MAE** is the average absolute error.
- **R²** is the fraction of variance explained.

Categorical columns are one-hot encoded and numeric ones standardized inside a `Pipeline`. Ridge (L2) and Lasso (L1) are also trained to show **regularization**.

## Files
`linear_regression.py`, `data/house_price.csv` (your lab dataset: Area, Bedrooms, Location, …, Price), `outputs/` (metrics, plots, model).

## Run
```bash
python linear_regression.py
```

## Result
The script prints the coefficients and compares the model against a mean-baseline and against Ridge and Lasso.

> **Observation:** this Kaggle house-price dataset was generated randomly, so its features have almost no real relationship with Price. That is why R² ≈ 0 and the model only matches the baseline. Explaining this in your viva shows real understanding. For a dataset with real signal, see **Experiment 16** (California Housing, R² ≈ 0.66).
