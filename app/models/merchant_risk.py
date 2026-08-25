import pandas as pd


INPUT_PATH = "data/synthetic/features.csv"
OUTPUT_PATH = "data/synthetic/features_with_merchant_risk.csv"


def add_historical_merchant_risk(df):
    df = df.copy()

    df["merchant_fraud_rate"] = 0.0

    # Process chronologically.
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Global prior prevents zero-information merchants
    # from getting an extreme score.
    global_rate = df["is_fraud"].mean()

    merchant_count = {}
    merchant_fraud = {}

    rates = []

    for _, row in df.iterrows():

        merchant = row["merchant"]

        count = merchant_count.get(merchant, 0)
        frauds = merchant_fraud.get(merchant, 0)

        # Historical rate BEFORE current transaction.
        if count == 0:
            rate = global_rate
        else:
            rate = frauds / count

        rates.append(rate)

        # Update state AFTER calculating current feature.
        merchant_count[merchant] = count + 1
        merchant_fraud[merchant] = (
            frauds + int(row["is_fraud"])
        )

    df["merchant_fraud_rate"] = rates

    return df


def main():

    print("Loading feature dataset...")

    df = pd.read_csv(INPUT_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    print("Calculating historical merchant risk...")

    df = add_historical_merchant_risk(df)

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nMerchant risk created.")

    print(
        df[
            [
                "merchant",
                "merchant_fraud_rate"
            ]
        ]
        .groupby("merchant")
        .agg(
            avg_historical_risk=(
                "merchant_fraud_rate",
                "mean"
            ),
            max_historical_risk=(
                "merchant_fraud_rate",
                "max"
            )
        )
        .sort_values(
            "avg_historical_risk",
            ascending=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()