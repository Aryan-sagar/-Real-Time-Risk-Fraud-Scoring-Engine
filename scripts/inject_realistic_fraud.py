import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

INPUT = "data/synthetic/transactions_sorted.csv"
OUTPUT = "data/synthetic/transactions_final.csv"

df = pd.read_csv(INPUT)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Reset existing fraud labels.
df["is_fraud"] = 0
df["fraud_type"] = None

# We want approximately 1.5% fraud.
FRAUD_COUNT = int(len(df) * 0.015)

# Five roughly balanced fraud categories.
fraud_types = (
    ["risky_merchant"] * 452
    + ["geo_velocity"] * 452
    + ["high_velocity"] * 452
    + ["account_takeover"] * 452
    + ["unusual_amount"] * 452
)

random.shuffle(fraud_types)

available_indices = list(df.index)
fraud_indices = random.sample(
    available_indices,
    FRAUD_COUNT
)


def mark_fraud(idx, fraud_type):
    df.loc[idx, "is_fraud"] = 1
    df.loc[idx, "fraud_type"] = fraud_type



# 1. RISKY MERCHANT

risky_indices = [
    idx for idx, fraud_type in zip(fraud_indices, fraud_types)
    if fraud_type == "risky_merchant"
]

for idx in risky_indices:
    mark_fraud(idx, "risky_merchant")

    df.loc[idx, "merchant"] = "Unknown Merchant"
    df.loc[idx, "category"] = "other"
    df.loc[idx, "merchant_base_risk"] = 0.08

    # Slightly elevated amount, but not absurdly large.
    df.loc[idx, "amount"] *= random.uniform(1.5, 4.0)



# 2. UNUSUAL AMOUNT

unusual_indices = [
    idx for idx, fraud_type in zip(fraud_indices, fraud_types)
    if fraud_type == "unusual_amount"
]

for idx in unusual_indices:
    account_id = df.loc[idx, "account_id"]

    account_history = df[
        (df["account_id"] == account_id)
        & (df["is_fraud"] == 0)
    ]

    if len(account_history) >= 3:
        normal_amount = account_history["amount"].median()
    else:
        normal_amount = df.loc[idx, "amount"]

    df.loc[idx, "amount"] = round(
        normal_amount * random.uniform(8, 25),
        2
    )

    mark_fraud(idx, "unusual_amount")



# 3. ACCOUNT TAKEOVER


takeover_indices = [
    idx for idx, fraud_type in zip(fraud_indices, fraud_types)
    if fraud_type == "account_takeover"
]

cities = list(
    df["city"].dropna().unique()
)

for idx in takeover_indices:

    account_id = df.loc[idx, "account_id"]

    account_history = df[
        (df["account_id"] == account_id)
        & (df["is_fraud"] == 0)
    ]

    if len(account_history) > 0:

        normal_amount = account_history["amount"].median()

        # Large deviation from normal behavior.
        df.loc[idx, "amount"] = round(
            normal_amount * random.uniform(3, 12),
            2
        )

        # Different city.
        current_city = df.loc[idx, "city"]

        alternative_cities = [
            city for city in cities
            if city != current_city
        ]

        new_city = random.choice(alternative_cities)

        df.loc[idx, "city"] = new_city

        # Update coordinates.
        city_coords = {
            "Delhi": (28.6139, 77.2090),
            "Mumbai": (19.0760, 72.8777),
            "Bangalore": (12.9716, 77.5946),
            "Hyderabad": (17.3850, 78.4867),
            "Chennai": (13.0827, 80.2707),
            "Kolkata": (22.5726, 88.3639),
            "Pune": (18.5204, 73.8567),
        }

        lat, lon = city_coords[new_city]

        df.loc[idx, "latitude"] = (
            lat + np.random.normal(0, 0.02)
        )

        df.loc[idx, "longitude"] = (
            lon + np.random.normal(0, 0.02)
        )

    mark_fraud(idx, "account_takeover")



# 4. GEO VELOCITY


geo_indices = [
    idx for idx, fraud_type in zip(fraud_indices, fraud_types)
    if fraud_type == "geo_velocity"
]

city_coords = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
}

cities = list(city_coords.keys())

for idx in geo_indices:

    account_id = df.loc[idx, "account_id"]

    previous = df[
        (df["account_id"] == account_id)
        & (df.index < idx)
    ].sort_values("timestamp")

    if len(previous) == 0:
        continue

    previous_transaction = previous.iloc[-1]

    previous_city = previous_transaction["city"]

    alternative_cities = [
        city for city in cities
        if city != previous_city
    ]

    new_city = random.choice(alternative_cities)

    # Force transaction to occur only a few minutes later.
    new_timestamp = (
        previous_transaction["timestamp"]
        + pd.Timedelta(
            minutes=random.randint(1, 5)
        )
    )

    df.loc[idx, "timestamp"] = new_timestamp
    df.loc[idx, "city"] = new_city

    lat, lon = city_coords[new_city]

    df.loc[idx, "latitude"] = (
        lat + np.random.normal(0, 0.02)
    )

    df.loc[idx, "longitude"] = (
        lon + np.random.normal(0, 0.02)
    )

    mark_fraud(idx, "geo_velocity")



# 5. HIGH VELOCITY


high_velocity_indices = [
    idx for idx, fraud_type in zip(fraud_indices, fraud_types)
    if fraud_type == "high_velocity"
]

for idx in high_velocity_indices:

    account_id = df.loc[idx, "account_id"]

    previous = df[
        (df["account_id"] == account_id)
        & (df.index < idx)
    ].sort_values("timestamp")

    if len(previous) == 0:
        continue

    previous_transaction = previous.iloc[-1]

    # Force this transaction to happen seconds/minutes later.
    new_timestamp = (
        previous_transaction["timestamp"]
        + pd.Timedelta(
            seconds=random.randint(10, 90)
        )
    )

    df.loc[idx, "timestamp"] = new_timestamp

    # Slightly varying amounts.
    previous_amount = previous_transaction["amount"]

    df.loc[idx, "amount"] = round(
        previous_amount * random.uniform(0.8, 1.5),
        2
    )

    mark_fraud(idx, "high_velocity")



# FINAL SORT

df = df.sort_values(
    ["account_id", "timestamp"]
).reset_index(drop=True)



# SAVE

df.to_csv(
    OUTPUT,
    index=False
)

print("Fraud pattern injection complete.")
print()
print("Dataset shape:", df.shape)
print()
print("Fraud distribution:")
print(df["is_fraud"].value_counts())
print()
print("Fraud types:")
print(
    df[df["is_fraud"] == 1]["fraud_type"]
    .value_counts()
)
print()
print("Fraud rate:")
print(
    f"{df['is_fraud'].mean() * 100:.2f}%"
)