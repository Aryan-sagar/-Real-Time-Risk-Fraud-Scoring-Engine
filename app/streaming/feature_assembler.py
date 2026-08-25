from datetime import datetime

from app.streaming.feature_state import FeatureState
from app.streaming.online_features import (
    calculate_online_features,
)


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


class FeatureAssembler:

    def __init__(self, feature_state: FeatureState):
        self.feature_state = feature_state

    def build(
        self,
        transaction,
        previous_transactions,
    ):
        account_stats = (
            self.feature_state.get_account_statistics(
                transaction["account_id"]
            )
        )

        online_features = (
            calculate_online_features(
                transaction,
                previous_transactions,
            )
        )

        amount = float(transaction["amount"])

        account_avg = float(
            account_stats["account_avg_amount"]
        )

        account_std = float(
            account_stats["account_std_amount"]
        )

        if account_avg == 0:
            amount_deviation = 0.0
        else:
            amount_deviation = (
                abs(amount - account_avg)
                / (account_std + 1.0)
            )

        timestamp = datetime.fromisoformat(
            transaction["timestamp"].replace(
                "Z",
                "+00:00",
            )
        )

        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = int(day_of_week >= 5)

        # Temporary fallback until live merchant-risk
        # state is implemented.
        merchant_fraud_rate = float(
            transaction["merchant_base_risk"]
        )

        features = {
            "amount": amount,
            "merchant_base_risk": float(
                transaction["merchant_base_risk"]
            ),
            "account_avg_amount": account_avg,
            "account_std_amount": account_std,
            "amount_deviation": amount_deviation,
            **online_features,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "merchant_fraud_rate": merchant_fraud_rate,
        }

        return {
            column: features[column]
            for column in FEATURE_COLUMNS
        }