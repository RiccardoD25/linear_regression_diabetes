# Linear Regression Diabetes Analysis

## Project Overview

This project develops and evaluates a multiple linear regression model using
Python and the diabetes dataset included with scikit-learn. The model predicts
a quantitative measure of diabetes disease progression one year after baseline
using 10 standardized clinical features.

The project was completed for the Week 3 assignment, **Develop and Analyze a
Linear Regression Model**.

## Dataset

The dataset contains:

- 442 patient observations
- 10 standardized baseline features
- A quantitative disease-progression target
- No missing values
- No duplicate observations
- No target outliers under the 1.5 × IQR rule

Dataset source:  
[Scikit-learn diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)

## Analysis Process

The Python program performs the following steps:

1. Loads and inspects the diabetes dataset.
2. Checks for missing values, duplicates, and potential outliers.
3. Reviews feature correlations with the target.
4. Splits the data into 80% training and 20% testing sets.
5. Trains an ordinary least squares linear regression model.
6. Evaluates the model using MAE, MSE, RMSE, and R-squared.
7. Interprets the model coefficients.
8. Conducts residual normality and variance checks.
9. Generates actual-versus-predicted, residual, Q-Q, and coefficient plots.

## Model Results

| Metric | Test Result |
|---|---:|
| R-squared | 0.453 |
| Mean Absolute Error | 42.794 |
| Mean Squared Error | 2,900.194 |
| Root Mean Squared Error | 53.853 |

The model explains approximately 45.3% of the variation in the unseen test
data. Residuals were approximately normal, but the analysis found evidence that
error variance may not remain constant across the prediction range.

## Repository File

- `linear_regression_diabetes.py` — complete data preparation, model training,
  evaluation, residual analysis, and visualization code.

## Requirements

- Python 3.10 or newer
- NumPy
- Pandas
- Matplotlib
- SciPy
- scikit-learn

Install the required packages with:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

## How to Run

Clone or download the repository, open a terminal in the project folder, and
run:

```bash
python linear_regression_diabetes.py
```

The program prints the dataset checks, correlations, model coefficients,
evaluation metrics, and residual diagnostics. It also creates a
`linear_regression_outputs` folder containing:

- `actual_vs_predicted.png`
- `residual_plot.png`
- `residual_qq_plot.png`
- `coefficients.png`
- `test_predictions_and_residuals.csv`

## Limitations

- Linear regression assumes additive linear relationships.
- Possible heteroscedasticity may reduce the reliability of standard errors and
  confidence intervals.
- Correlated serum measures may make individual coefficients unstable.
- The results show associations and should not be interpreted as causal.
- The model is an educational baseline and is not intended for clinical use.

## Author

Riccardo De Simini
