import os

import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier


DATA_PATH = "data/processed/test_set.csv"
MODEL_PATH = "data/processed/fraud_xgboost.json"
OUTPUT_DIR = "data/processed/shap"


FEATURE_COLUMNS = [
    "amount",
    "merchant_base_risk",
    "account_avg_amount",
    "account_std_amount",
    "amount_deviation",
    "txn_count_1m",
    "txn_count_10m",
    "txn_count_1h",
    "amount_sum_10m",
    "amount_sum_1h",
    "distance_from_previous",
    "time_since_previous_minutes",
    "geo_velocity_kmh",
    "hour",
    "day_of_week",
    "is_weekend",
    "merchant_fraud_rate",
]


def main():

    print("Loading test data...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]

    print("Loading XGBoost model...")

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    print("Creating SHAP explainer...")

    explainer = shap.TreeExplainer(model)

    print("Calculating SHAP values...")

    # Use a representative sample rather than the
    # entire test set for the global analysis.
    sample_size = min(5000, len(X))

    X_sample = X.sample(
        sample_size,
        random_state=42
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Global feature importance
    # ---------------------------------------------------------

    mean_abs_shap = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "mean_abs_shap": (
                    abs(shap_values)
                    .mean(axis=0)
                ),
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("GLOBAL SHAP FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        mean_abs_shap
        .to_string(index=False)
    )

    mean_abs_shap.to_csv(
        f"{OUTPUT_DIR}/global_importance.csv",
        index=False
    )

    # ---------------------------------------------------------
    # SHAP summary plot
    # ---------------------------------------------------------

    print("\nCreating SHAP summary plot...")

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/summary_plot.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # Explain individual fraud cases
    # ---------------------------------------------------------

    probabilities = model.predict_proba(
        X_sample
    )[:, 1]

    X_sample_with_score = X_sample.copy()

    X_sample_with_score["risk_probability"] = (
        probabilities
    )

    highest_risk_indices = (
        X_sample_with_score[
            "risk_probability"
        ]
        .nlargest(5)
        .index
    )

    print("\n" + "=" * 70)
    print("HIGH-RISK TRANSACTION EXPLANATIONS")
    print("=" * 70)

    for idx in highest_risk_indices:

        row_position = X_sample.index.get_loc(idx)

        explanation = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "feature_value": X_sample.loc[
                    idx
                ].values,
                "shap_value": shap_values[
                    row_position
                ],
            }
        )

        explanation["abs_shap"] = (
            explanation["shap_value"]
            .abs()
        )

        explanation = (
            explanation
            .sort_values(
                "abs_shap",
                ascending=False
            )
            .head(5)
        )

        print(
            f"\nTransaction index: {idx}"
        )

        print(
            f"Risk probability: "
            f"{probabilities[row_position]:.4f}"
        )

        print(
            explanation[
                [
                    "feature",
                    "feature_value",
                    "shap_value",
                ]
            ].to_string(index=False)
        )

    print("\nSHAP analysis complete.")

    print(
        f"\nSaved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()