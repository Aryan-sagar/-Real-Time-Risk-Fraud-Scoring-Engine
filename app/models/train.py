import json
import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from xgboost import XGBClassifier


INPUT_PATH = "data/synthetic/features_with_merchant_risk.csv"
OUTPUT_DIR = "data/processed"


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
]


def add_merchant_historical_risk(df):
    """
    Calculate merchant fraud rate using only historical
    transactions.

    IMPORTANT:
    The current transaction is excluded.

    IMPORTANT:
    This assumes `df` is already sorted chronologically. shift(1)
    + expanding() only means "everything before this row in time"
    if rows are actually in time order going in.
    """

    df = df.copy()

    df["merchant_fraud_rate"] = 0.0

    for merchant, group in df.groupby("merchant_base_risk"):
        # We use the existing merchant risk as a stable grouping
        # signal for now. The actual merchant identifier is not
        # present in features.csv, so this remains a lightweight
        # risk proxy.
        historical_fraud_rate = (
            group["is_fraud"]
            .shift(1)
            .expanding()
            .mean()
        )

        df.loc[group.index, "merchant_fraud_rate"] = (
            historical_fraud_rate
        )

    df["merchant_fraud_rate"] = (
        df["merchant_fraud_rate"]
        .fillna(0)
    )

    return df


def chronological_split(df):
    """
    Split data chronologically.

    70% train
    15% validation
    15% test

    Note: `df` is expected to already be sorted by timestamp by the
    time it gets here (see train()) — this re-sort is just a cheap
    safety net, not the only place ordering is enforced.
    """

    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    return train, validation, test


def find_best_threshold(y_true, probabilities):
    """
    Pick the probability threshold that maximizes F1 on this set.

    Only ever call this on validation. Apply the resulting threshold
    as-is to test — re-deriving it per split defeats the point of
    having a held-out set.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    # precision/recall have one more point than thresholds (the last
    # point is threshold=inf, recall=0), so drop it before comparing.
    precision, recall = precision[:-1], recall[:-1]

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    best_idx = np.argmax(f1)

    return thresholds[best_idx], f1[best_idx]


def evaluate_model(model, X, y, name, threshold=0.5):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")

    print(f"Threshold: {threshold:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": threshold,
    }


def train():
    print("Loading feature dataset...")

    df = pd.read_csv(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Sort chronologically BEFORE any "historical" feature is derived.
    # add_merchant_historical_risk() uses shift(1).expanding() per
    # merchant group, which only reflects "the past" if the rows are
    # already in time order when it runs. Doing this sort later (e.g.
    # only inside chronological_split) lets that feature leak
    # future fraud labels into earlier rows.
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Rows: {len(df):,}")

    
    # Merchant risk
    

    print("\nCreating historical merchant risk...")

    # features.csv currently doesn't contain merchant name.
    # We'll use merchant_base_risk as a temporary proxy.
    df = add_merchant_historical_risk(df)

    FEATURE_COLUMNS_WITH_RISK = (
        FEATURE_COLUMNS
        + ["merchant_fraud_rate"]
    )

    
    # Chronological split
    

    print("\nCreating chronological split...")

    train_df, validation_df, test_df = chronological_split(df)

    print(
        f"Train      : {len(train_df):,}"
    )
    print(
        f"Validation : {len(validation_df):,}"
    )
    print(
        f"Test       : {len(test_df):,}"
    )

    print("\nFraud rate:")
    print(
        f"Train      : {train_df['is_fraud'].mean() * 100:.3f}%"
    )
    print(
        f"Validation : {validation_df['is_fraud'].mean() * 100:.3f}%"
    )
    print(
        f"Test       : {test_df['is_fraud'].mean() * 100:.3f}%"
    )

    
    # Prepare matrices
    

    X_train = train_df[
        FEATURE_COLUMNS_WITH_RISK
    ].copy()

    y_train = train_df["is_fraud"]

    X_validation = validation_df[
        FEATURE_COLUMNS_WITH_RISK
    ].copy()

    y_validation = validation_df["is_fraud"]

    X_test = test_df[
        FEATURE_COLUMNS_WITH_RISK
    ].copy()

    y_test = test_df["is_fraud"]

    # Safety check.
    assert not X_train.isnull().any().any()
    assert not X_validation.isnull().any().any()
    assert not X_test.isnull().any().any()

    
    # Handle class imbalance
    

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    if positive_count == 0:
        raise ValueError(
            "No positive (fraud) examples in the training split — "
            "check the source data and chronological_split() before "
            "training on this."
        )

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        f"\nscale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    
    # XGBoost
    

    print("\nTraining XGBoost...")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=False,
    )

    
    # Threshold selection (validation only)
    

    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    best_threshold, best_f1 = find_best_threshold(
        y_validation, validation_probabilities
    )

    print(
        f"\nSelected threshold from validation PR curve: "
        f"{best_threshold:.4f} (F1={best_f1:.4f})"
    )

    
    # Evaluation
    

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        "VALIDATION",
        threshold=best_threshold,
    )

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        "TEST",
        threshold=best_threshold,
    )

    
    # Feature importance
    
    importance = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS_WITH_RISK,
                "importance": model.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print("\nFeature importance:")
    print(importance.to_string(index=False))

    
    # Save model + split datasets
    
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    model.save_model(
        f"{OUTPUT_DIR}/fraud_xgboost.json"
    )

    # The threshold was deliberately derived from validation, not
    # hardcoded — persist it so scoring code downstream (e.g.
    # FraudModelScorer) uses the calibrated value instead of falling
    # back to an arbitrary default.
    metadata = {
        "threshold": float(best_threshold),
        "validation_f1_at_threshold": float(best_f1),
        "validation_pr_auc": float(validation_metrics["pr_auc"]),
        "validation_roc_auc": float(validation_metrics["roc_auc"]),
        "test_pr_auc": float(test_metrics["pr_auc"]),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "feature_columns": FEATURE_COLUMNS_WITH_RISK,
    }

    with open(f"{OUTPUT_DIR}/fraud_xgboost.metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    train_df.to_csv(
        f"{OUTPUT_DIR}/train_set.csv",
        index=False
    )

    test_df.to_csv(
        f"{OUTPUT_DIR}/test_set.csv",
        index=False
    )

    validation_df.to_csv(
        f"{OUTPUT_DIR}/validation_set.csv",
        index=False
    )

    print("\nSaved:")
    print(
        f"{OUTPUT_DIR}/fraud_xgboost.json"
    )
    print(
        f"{OUTPUT_DIR}/fraud_xgboost.metadata.json"
    )
    print(
        f"{OUTPUT_DIR}/train_set.csv"
    )
    print(
        f"{OUTPUT_DIR}/test_set.csv"
    )
    print(
        f"{OUTPUT_DIR}/validation_set.csv"
    )


if __name__ == "__main__":
    train()