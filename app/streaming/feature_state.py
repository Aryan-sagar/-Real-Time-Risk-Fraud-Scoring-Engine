import json
import os
from datetime import datetime, timezone

import redis


REDIS_HOST = os.getenv(
    "REDIS_HOST",
    "localhost",
)

REDIS_PORT = int(
    os.getenv(
        "REDIS_PORT",
        "6379",
    )
)

PROCESSED_SET = "processed_transactions"


def parse_timestamp(value):
    value = str(value)

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    parsed = datetime.fromisoformat(value)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


class FeatureState:

    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
        )

    def account_key(self, account_id):
        return f"account:{account_id}"

    # ---------------------------------------------------------
    # Idempotency
    # ---------------------------------------------------------

    def transaction_processed(self, transaction_id):
        return self.redis.sismember(
            PROCESSED_SET,
            transaction_id,
        )

    def mark_transaction_processed(self, transaction_id):
        self.redis.sadd(
            PROCESSED_SET,
            transaction_id,
        )

    # ---------------------------------------------------------
    # Account state
    # ---------------------------------------------------------

    def get_account_state(self, account_id):
        raw_state = self.redis.get(
            self.account_key(account_id)
        )

        if raw_state is None:
            return {
                "transactions": [],
                "transaction_count": 0,
                "total_amount": 0.0,
                "historical_amounts": [],
            }

        state = json.loads(raw_state)

        state.setdefault("transactions", [])
        state.setdefault("historical_amounts", [])

        state["transaction_count"] = len(
            state["transactions"]
        )

        state["total_amount"] = sum(
            float(tx["amount"])
            for tx in state["transactions"]
        )

        return state

    def save_account_state(
        self,
        account_id,
        state,
    ):
        state["transaction_count"] = len(
            state["transactions"]
        )

        state["total_amount"] = sum(
            float(tx["amount"])
            for tx in state["transactions"]
        )

        self.redis.set(
            self.account_key(account_id),
            json.dumps(state),
            ex=86400,
        )

    # ---------------------------------------------------------
    # Recent transactions
    # ---------------------------------------------------------

    def get_recent_transactions(
        self,
        account_id,
        current_timestamp,
    ):
        state = self.get_account_state(
            account_id
        )

        current_time = parse_timestamp(
            current_timestamp
        )

        recent_transactions = []

        for transaction in state["transactions"]:
            transaction_time = parse_timestamp(
                transaction["timestamp"]
            )

            age_seconds = (
                current_time - transaction_time
            ).total_seconds()

            if 0 <= age_seconds <= 3600:
                recent_transactions.append(
                    transaction
                )

        return recent_transactions

    # ---------------------------------------------------------
    # Historical account statistics
    # ---------------------------------------------------------

    def get_account_statistics(self, account_id):
        state = self.get_account_state(
            account_id
        )

        amounts = [
            float(amount)
            for amount in state["historical_amounts"]
        ]

        if not amounts:
            return {
                "account_avg_amount": 0.0,
                "account_std_amount": 0.0,
            }

        if len(amounts) == 1:
            return {
                "account_avg_amount": amounts[0],
                "account_std_amount": 0.0,
            }

        mean_amount = sum(amounts) / len(amounts)

        variance = sum(
            (amount - mean_amount) ** 2
            for amount in amounts
        ) / (len(amounts) - 1)

        return {
            "account_avg_amount": mean_amount,
            "account_std_amount": variance ** 0.5,
        }

    # ---------------------------------------------------------
    # Add transaction
    # ---------------------------------------------------------

    def add_transaction(
        self,
        account_id,
        timestamp,
        amount,
        latitude,
        longitude,
        transaction_id=None,
        merchant="Unknown Merchant",
        merchant_base_risk=0.0,
        is_fraud=None,
    ):
        if transaction_id is None:
            transaction_id = (
                f"TEST_{account_id}_{timestamp}"
            )
        if self.transaction_processed(
            transaction_id
        ):
            return None

        state = self.get_account_state(
            account_id
        )
    

        transaction = {
            "transaction_id": transaction_id,
            "timestamp": str(timestamp),
            "amount": float(amount),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "merchant": merchant,
            "merchant_base_risk": float(
                merchant_base_risk
            ),
        }

        state["transactions"].append(
            transaction
        )

        # Historical amount state.
        state["historical_amounts"].append(
            float(amount)
        )

        current_time = parse_timestamp(
            timestamp
        )

        # Keep only the previous hour in online state.
        filtered_transactions = []

        for tx in state["transactions"]:
            tx_time = parse_timestamp(
                tx["timestamp"]
            )

            age_seconds = (
                current_time - tx_time
            ).total_seconds()

            if 0 <= age_seconds <= 3600:
                filtered_transactions.append(
                    tx
                )

        state["transactions"] = (
            filtered_transactions
        )

        self.save_account_state(
            account_id,
            state,
        )

        self.mark_transaction_processed(
            transaction_id
        )

        return state