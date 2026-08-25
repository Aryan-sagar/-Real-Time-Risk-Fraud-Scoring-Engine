from app.streaming.online_features import (
    calculate_online_features,
)


def test_online_velocity_features():

    previous_transactions = [
        {
            "timestamp": "2026-08-25T03:00:00+00:00",
            "amount": 500.0,
            "latitude": 28.6139,
            "longitude": 77.2090,
        },
        {
            "timestamp": "2026-08-25T03:05:00+00:00",
            "amount": 300.0,
            "latitude": 28.6200,
            "longitude": 77.2100,
        },
    ]

    current_transaction = {
        "timestamp": "2026-08-25T03:05:30+00:00",
        "amount": 700.0,
        "latitude": 28.6300,
        "longitude": 77.2200,
    }

    features = calculate_online_features(
        current_transaction,
        previous_transactions,
    )

    # One transaction occurred during the
    # previous minute.
    assert features["txn_count_1m"] == 1

    # Both previous transactions occurred
    # during the previous 10 minutes.
    assert features["txn_count_10m"] == 2

    # Both also occurred during the previous hour.
    assert features["txn_count_1h"] == 2

    # Previous transaction amounts.
    assert features["amount_sum_10m"] == 800.0
    assert features["amount_sum_1h"] == 800.0

    # There must be a previous location.
    assert features["distance_from_previous"] > 0

    # 30 seconds since previous transaction.
    assert (
        features["time_since_previous_minutes"]
        == 0.5
    )

    # Geographic velocity should be positive.
    assert features["geo_velocity_kmh"] > 0