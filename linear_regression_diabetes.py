"""
Week 3 Assignment: Develop and Analyze a Linear Regression Model

This script develops and evaluates a multiple linear regression model using
scikit-learn's diabetes dataset. It documents preparation, feature selection,
training, performance evaluation, coefficient interpretation, and residual
analysis. Running the script creates three PNG visualizations and a CSV file
containing test-set predictions and residuals.

Dataset source:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20
OUTPUT_DIR = Path(__file__).resolve().parent / "linear_regression_outputs"


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset, verify quality, and return predictors and target."""
    diabetes = load_diabetes(as_frame=True, scaled=True)
    data = diabetes.frame.copy()

    print("Dataset shape:", data.shape)
    print("Missing values:", int(data.isna().sum().sum()))
    print("Duplicate rows:", int(data.duplicated().sum()))

    # The dataset has no missing values. Duplicate baseline feature rows can be
    # legitimate patients, so observations are retained rather than deleted.
    # Potential target outliers are reviewed using the 1.5 × IQR rule.
    q1, q3 = data["target"].quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_mask = (data["target"] < q1 - 1.5 * iqr) | (
        data["target"] > q3 + 1.5 * iqr
    )
    print("Potential target outliers (1.5 x IQR):", int(outlier_mask.sum()))
    print("Outlier decision: retain clinically plausible observations.")

    # All 10 baseline features are retained. They are standardized by the
    # dataset provider and have established clinical relevance. Correlations
    # are reported to support interpretation rather than used to cherry-pick.
    correlations = (
        data.corr(numeric_only=True)["target"]
        .drop("target")
        .sort_values(key=abs, ascending=False)
    )
    print("\nFeature correlations with target:\n", correlations.round(3))

    return data.drop(columns="target"), data["target"]


def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series
) -> tuple[LinearRegression, pd.DataFrame, dict[str, float]]:
    """Split the data, train OLS regression, and calculate evaluation metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    residuals = y_test.to_numpy() - predictions

    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, predictions),
    }

    results = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Predicted": predictions,
            "Residual": residuals,
        },
        index=y_test.index,
    ).sort_index()

    print(f"\nTraining observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")
    print("Intercept:", round(float(model.intercept_), 3))
    print("\nTest metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    coefficients = pd.Series(model.coef_, index=X.columns).sort_values()
    print("\nCoefficients:\n", coefficients.round(3))

    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    bp_correlation, bp_p = stats.pearsonr(np.abs(residuals), predictions)
    print("\nResidual diagnostics:")
    print(f"Mean residual: {residuals.mean():.3f}")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(
        "Absolute-residual/prediction correlation "
        f"(heteroscedasticity screen): r={bp_correlation:.3f}, p={bp_p:.4f}"
    )

    return model, results, metrics


def create_visualizations(
    model: LinearRegression,
    feature_names: pd.Index,
    results: pd.DataFrame,
) -> None:
    """Create actual-vs-predicted, residual, Q-Q, and coefficient plots."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(
        results["Actual"], results["Predicted"], alpha=0.72, color="#2563EB"
    )
    lower = min(results["Actual"].min(), results["Predicted"].min())
    upper = max(results["Actual"].max(), results["Predicted"].max())
    plt.plot([lower, upper], [lower, upper], "--", color="#F97316", linewidth=2)
    plt.xlabel("Actual disease-progression score")
    plt.ylabel("Predicted score")
    plt.title("Actual vs. Predicted Values")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(
        results["Predicted"], results["Residual"], alpha=0.72, color="#2563EB"
    )
    plt.axhline(0, linestyle="--", color="#F97316", linewidth=2)
    plt.xlabel("Predicted score")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residuals vs. Predicted Values")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    stats.probplot(results["Residual"], dist="norm", plot=plt)
    plt.title("Normal Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_qq_plot.png", dpi=180)
    plt.close()

    coefficient_series = pd.Series(model.coef_, index=feature_names).sort_values()
    colors = ["#F97316" if value < 0 else "#2563EB" for value in coefficient_series]
    plt.figure(figsize=(8, 5))
    coefficient_series.plot(kind="barh", color=colors)
    plt.axvline(0, color="#111827", linewidth=1)
    plt.xlabel("Coefficient (target-score units per standardized feature unit)")
    plt.title("Linear Regression Coefficients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coefficients.png", dpi=180)
    plt.close()


def main() -> None:
    """Run the complete linear regression workflow."""
    X, y = load_and_prepare_data()
    model, results, _ = train_and_evaluate(X, y)
    create_visualizations(model, X.columns, results)
    results.to_csv(OUTPUT_DIR / "test_predictions_and_residuals.csv", index=True)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
