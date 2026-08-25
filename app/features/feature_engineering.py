import numpy as np
import pandas as pd


INPUT_PATH = "data/synthetic/transactions_final.csv"
OUTPUT_PATH = "data/synthetic/features.csv"


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two geographic coordinates in kilometers.
    """
    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def create_account_features(df):
    """
    Create historical account-behavior features.

    IMPORTANT:
    These statistics are shifted so the current transaction
    cannot use its own value.
    """

    grouped = df.groupby("account_id")

    df["account_avg_amount"] = (
        grouped["amount"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df["account_std_amount"] = (
        grouped["amount"]
        .transform(lambda x: x.shift(1).expanding().std())
    )

    # For the first transaction of an account,
    # fall back to the current amount rather than NaN.
    df["account_avg_amount"] = (
        df["account_avg_amount"]
        .fillna(df["amount"])
    )

    df["account_std_amount"] = (
        df["account_std_amount"]
        .fillna(0)
    )

    df["amount_deviation"] = (
        abs(df["amount"] - df["account_avg_amount"])
        / (df["account_std_amount"] + 1)
    )

    return df


def create_velocity_features(df):
    """
    Calculate transaction velocity using only previous
    transactions from the same account.
    """

    df["txn_count_1m"] = 0.0
    df["txn_count_10m"] = 0.0
    df["txn_count_1h"] = 0.0

    df["amount_sum_10m"] = 0.0
    df["amount_sum_1h"] = 0.0

    for account_id, group in df.groupby("account_id"):

        group = group.sort_values("timestamp")

        timestamps = group["timestamp"]
        amounts = group["amount"]

        rolling_base = pd.Series(
            amounts.values,
            index=timestamps
        )

        count_1m = (
            rolling_base
            .rolling("1min", closed="left")
            .count()
            .fillna(0)
        )

        count_10m = (
            rolling_base
            .rolling("10min", closed="left")
            .count()
            .fillna(0)
        )

        count_1h = (
            rolling_base
            .rolling("1h", closed="left")
            .count()
            .fillna(0)
        )

        amount_10m = (
            rolling_base
            .rolling("10min", closed="left")
            .sum()
            .fillna(0)
        )

        amount_1h = (
            rolling_base
            .rolling("1h", closed="left")
            .sum()
            .fillna(0)
        )

        df.loc[group.index, "txn_count_1m"] = (
            count_1m.to_numpy()
        )

        df.loc[group.index, "txn_count_10m"] = (
            count_10m.to_numpy()
        )

        df.loc[group.index, "txn_count_1h"] = (
            count_1h.to_numpy()
        )

        df.loc[group.index, "amount_sum_10m"] = (
            amount_10m.to_numpy()
        )

        df.loc[group.index, "amount_sum_1h"] = (
            amount_1h.to_numpy()
        )

    return df


def create_geo_features(df):
    """
    Calculate distance and geographic velocity from
    the previous transaction.
    """

    df["previous_latitude"] = (
        df.groupby("account_id")["latitude"]
        .shift(1)
    )

    df["previous_longitude"] = (
        df.groupby("account_id")["longitude"]
        .shift(1)
    )

    df["previous_timestamp"] = (
        df.groupby("account_id")["timestamp"]
        .shift(1)
    )

    df["distance_from_previous"] = haversine_distance(
        df["previous_latitude"],
        df["previous_longitude"],
        df["latitude"],
        df["longitude"],
    )

    df["time_since_previous_minutes"] = (
        (
            df["timestamp"]
            - df["previous_timestamp"]
        )
        .dt.total_seconds()
        / 60
    )

    df["geo_velocity_kmh"] = (
        df["distance_from_previous"]
        / (df["time_since_previous_minutes"] / 60)
    )

    df["distance_from_previous"] = (
        df["distance_from_previous"].fillna(0)
    )

    df["time_since_previous_minutes"] = (
        df["time_since_previous_minutes"].fillna(-1)
    )

    df["geo_velocity_kmh"] = (
        df["geo_velocity_kmh"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return df


def create_time_features(df):
    """
    Extract time-based behavioral features.
    """

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


def build_features():

    print("Loading transactions...")

    df = pd.read_csv(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # Critical: chronological ordering.
    df = (
        df.sort_values(
            ["account_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    print("Creating account features...")
    df = create_account_features(df)

    print("Creating velocity features...")
    df = create_velocity_features(df)

    print("Creating geographic features...")
    df = create_geo_features(df)

    print("Creating time features...")
    df = create_time_features(df)

    # Features that the ML model is allowed to see.
    feature_columns = [
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

    # Keep labels separately.
    output_columns = (
        ["transaction_id", 
         "account_id",
         "timestamp",
         "merchant"]
        + feature_columns
        + ["is_fraud", "fraud_type"]
    )

    features = df[output_columns].copy()

    features.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFeature engineering complete.")
    print(f"Rows: {len(features):,}")
    print(f"Features: {len(feature_columns)}")

    print("\nFeature columns:")
    for column in feature_columns:
        print(f"  - {column}")

    print("\nMissing values:")
    print(
        features[feature_columns]
        .isnull()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_features()