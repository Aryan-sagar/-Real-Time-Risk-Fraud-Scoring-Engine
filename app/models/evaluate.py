import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
)


DATA_PATH = "data/processed/test_set.csv"
MODEL_PATH = "data/processed/fraud_xgboost.json"


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


def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def evaluate_thresholds(y_true, probabilities):

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
    ]

    results = []

    total_legitimate = (y_true == 0).sum()

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions
        ).ravel()

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        false_positive_rate = (
            fp / total_legitimate
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "false_positive_rate": false_positive_rate,
                "false_positives": fp,
                "false_negatives": fn,
            }
        )

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "precision": "{:.3f}".format,
                "recall": "{:.3f}".format,
                "false_positive_rate": "{:.4f}".format,
            },
        )
    )

    return results_df


def analyze_errors(df, probabilities):

    df = df.copy()

    df["risk_probability"] = probabilities

    df["prediction"] = (
        df["risk_probability"] >= 0.50
    ).astype(int)

    
    # False positives
    

    false_positives = df[
        (df["prediction"] == 1)
        & (df["is_fraud"] == 0)
    ]

    print("\n" + "=" * 70)
    print("FALSE POSITIVES")
    print("=" * 70)

    print(
        f"Count: {len(false_positives):,}"
    )

    print("\nHighest-risk false positives:")

    fp_columns = [
        "transaction_id",
        "amount",
        "amount_deviation",
        "txn_count_10m",
        "geo_velocity_kmh",
        "merchant_base_risk",
        "risk_probability",
    ]

    print(
        false_positives
        .sort_values(
            "risk_probability",
            ascending=False
        )[fp_columns]
        .head(15)
        .to_string(index=False)
    )

    
    # False negatives
    
    false_negatives = df[
        (df["prediction"] == 0)
        & (df["is_fraud"] == 1)
    ]

    print("\n" + "=" * 70)
    print("FALSE NEGATIVES")
    print("=" * 70)

    print(
        f"Count: {len(false_negatives):,}"
    )

    fn_columns = [
        "transaction_id",
        "fraud_type",
        "amount",
        "amount_deviation",
        "txn_count_10m",
        "geo_velocity_kmh",
        "merchant_base_risk",
        "risk_probability",
    ]

    print(
        false_negatives
        .sort_values(
            "risk_probability"
        )[fn_columns]
        .head(15)
        .to_string(index=False)
    )

    
    # Fraud type performance
    
    fraud_only = df[
        df["is_fraud"] == 1
    ].copy()

    fraud_only["prediction"] = (
        fraud_only["risk_probability"] >= 0.50
    ).astype(int)

    print("\n" + "=" * 70)
    print("RECALL BY FRAUD TYPE")
    print("=" * 70)

    type_results = []

    for fraud_type, group in fraud_only.groupby(
        "fraud_type"
    ):

        recall = recall_score(
            group["is_fraud"],
            group["prediction"],
            zero_division=0
        )

        type_results.append(
            {
                "fraud_type": fraud_type,
                "count": len(group),
                "recall": recall,
            }
        )

    print(
        pd.DataFrame(type_results)
        .sort_values("recall")
        .to_string(index=False)
    )

    return df


def main():

    print("Loading test dataset...")

    df = pd.read_csv(DATA_PATH)

    model = load_model()

    X = df[
        FEATURE_COLUMNS
    ]

    y = df["is_fraud"]

    probabilities = model.predict_proba(X)[:, 1]

    print(
        f"\nTest PR-AUC: "
        f"{average_precision_score(y, probabilities):.4f}"
    )

    threshold_results = evaluate_thresholds(
        y,
        probabilities
    )

    analyzed_df = analyze_errors(
        df,
        probabilities
    )

    threshold_results.to_csv(
        "data/processed/threshold_analysis.csv",
        index=False
    )

    analyzed_df.to_csv(
        "data/processed/test_predictions.csv",
        index=False
    )

    print("\nSaved:")
    print(
        "data/processed/threshold_analysis.csv"
    )
    print(
        "data/processed/test_predictions.csv"
    )


if __name__ == "__main__":
    main()