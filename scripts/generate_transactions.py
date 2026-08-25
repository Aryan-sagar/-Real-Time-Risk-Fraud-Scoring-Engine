import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker


fake = Faker()
random.seed(42)
np.random.seed(42)

N_USERS = 10_000
N_TRANSACTIONS = 150_000
FRAUD_RATE = 0.015

CITIES = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
}

MERCHANTS = [
    ("Amazon", "shopping", 0.01),
    ("Flipkart", "shopping", 0.015),
    ("Swiggy", "food", 0.005),
    ("Zomato", "food", 0.005),
    ("Uber", "transport", 0.003),
    ("Netflix", "entertainment", 0.002),
    ("BookMyShow", "entertainment", 0.004),
    ("BigBasket", "grocery", 0.008),
    ("Myntra", "shopping", 0.012),
    ("Croma", "electronics", 0.02),
    ("Unknown Merchant", "other", 0.08),
]


def generate_users():
    users = []

    for user_id in range(1, N_USERS + 1):
        home_city = random.choice(list(CITIES.keys()))

        users.append(
            {
                "account_id": f"ACC_{user_id:06d}",
                "home_city": home_city,
                "avg_amount": round(np.random.lognormal(5.3, 0.55), 2),
                "monthly_income": round(
                    np.random.lognormal(10.5, 0.55), 2
                ),
            }
        )

    return pd.DataFrame(users)


def choose_merchant():
    names = [m[0] for m in MERCHANTS]
    weights = [1 / (m[2] + 0.01) for m in MERCHANTS]

    return random.choices(names, weights=weights, k=1)[0]


def merchant_info(name):
    for merchant in MERCHANTS:
        if merchant[0] == name:
            return merchant
    return MERCHANTS[-1]


def generate_transactions(users):
    transactions = []

    start_time = datetime(2026, 1, 1)

    for transaction_id in range(1, N_TRANSACTIONS + 1):

        user = users.sample(1).iloc[0]

        merchant = choose_merchant()
        _, category, base_risk = merchant_info(merchant)

        amount = max(
            20,
            np.random.lognormal(
                np.log(user["avg_amount"]),
                0.65
            )
        )

        timestamp = start_time + timedelta(
            minutes=random.randint(0, 180 * 24 * 60)
        )

        city = user["home_city"]

        is_fraud = random.random() < FRAUD_RATE

        fraud_type = None

        if is_fraud:
            fraud_type = random.choice(
                [
                    "unusual_amount",
                    "geo_velocity",
                    "high_velocity",
                    "risky_merchant",
                    "account_takeover",
                ]
            )

            if fraud_type == "unusual_amount":
                amount *= random.uniform(8, 30)

            elif fraud_type == "geo_velocity":
                city = random.choice(
                    [
                        c for c in CITIES.keys()
                        if c != user["home_city"]
                    ]
                )

            elif fraud_type == "high_velocity":
                pass

            elif fraud_type == "risky_merchant":
                merchant = "Unknown Merchant"
                category = "other"

            elif fraud_type == "account_takeover":
                amount *= random.uniform(3, 15)
                city = random.choice(list(CITIES.keys()))

        lat, lon = CITIES[city]

        transactions.append(
            {
                "transaction_id": f"TXN_{transaction_id:09d}",
                "account_id": user["account_id"],
                "merchant": merchant,
                "category": category,
                "amount": round(amount, 2),
                "currency": "INR",
                "city": city,
                "latitude": lat + np.random.normal(0, 0.02),
                "longitude": lon + np.random.normal(0, 0.02),
                "timestamp": timestamp,
                "merchant_base_risk": base_risk,
                "is_fraud": int(is_fraud),
                "fraud_type": fraud_type,
            }
        )

    return pd.DataFrame(transactions)


def inject_duplicates(df, fraction=0.002):
    duplicate_count = int(len(df) * fraction)

    duplicates = df.sample(
        duplicate_count,
        random_state=42
    ).copy()

    return pd.concat(
        [df, duplicates],
        ignore_index=True
    )


def inject_bad_records(df, fraction=0.001):
    bad_count = int(len(df) * fraction)

    indices = np.random.choice(
        df.index,
        bad_count,
        replace=False
    )

    df.loc[indices, "amount"] = -1

    return df


def main():
    print("Generating users...")
    users = generate_users()

    print("Generating transactions...")
    transactions = generate_transactions(users)

    print("Injecting duplicate records...")
    transactions = inject_duplicates(transactions)

    print("Injecting bad records...")
    transactions = inject_bad_records(transactions)

    os.makedirs("data/synthetic", exist_ok=True)

    users.to_csv(
        "data/synthetic/users.csv",
        index=False
    )

    transactions.to_csv(
        "data/synthetic/transactions.csv",
        index=False
    )

    print("\nGeneration complete.")
    print(f"Users: {len(users):,}")
    print(f"Transactions: {len(transactions):,}")
    print(
        f"Fraud transactions: "
        f"{transactions['is_fraud'].sum():,}"
    )

    print("\nFraud distribution:")
    print(
        transactions[
            transactions["is_fraud"] == 1
        ]["fraud_type"].value_counts()
    )

    print("\nFiles created:")
    print("data/synthetic/users.csv")
    print("data/synthetic/transactions.csv")


if __name__ == "__main__":
    main()