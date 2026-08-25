import math


def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2,
):
    R = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.asin(
        math.sqrt(a)
    )


def calculate_online_features(
    current_transaction,
    previous_transactions,
):
    current_timestamp = (
        current_transaction["timestamp"]
    )

    current_amount = float(
        current_transaction["amount"]
    )

    current_lat = float(
        current_transaction["latitude"]
    )

    current_lon = float(
        current_transaction["longitude"]
    )

    features = {}

    # ---------------------------------------------------------
    # Velocity
    # ---------------------------------------------------------

    def seconds_before(tx):

        from datetime import datetime

        current = datetime.fromisoformat(
            current_timestamp.replace("Z", "+00:00")
        )

        previous = datetime.fromisoformat(
            tx["timestamp"].replace("Z", "+00:00")
        )

        return (
            current - previous
        ).total_seconds()

    last_1m = [
        tx for tx in previous_transactions
        if seconds_before(tx) <= 60
    ]

    last_10m = [
        tx for tx in previous_transactions
        if seconds_before(tx) <= 600
    ]

    last_1h = [
        tx for tx in previous_transactions
        if seconds_before(tx) <= 3600
    ]

    features["txn_count_1m"] = len(
        last_1m
    )

    features["txn_count_10m"] = len(
        last_10m
    )

    features["txn_count_1h"] = len(
        last_1h
    )

    features["amount_sum_10m"] = sum(
        tx["amount"]
        for tx in last_10m
    )

    features["amount_sum_1h"] = sum(
        tx["amount"]
        for tx in last_1h
    )

    # ---------------------------------------------------------
    # Previous transaction
    # ---------------------------------------------------------

    if previous_transactions:

        previous = previous_transactions[-1]

        distance = haversine_distance(
            previous["latitude"],
            previous["longitude"],
            current_lat,
            current_lon,
        )

        time_seconds = seconds_before(
            previous
        )

        time_minutes = (
            time_seconds / 60
        )

        features[
            "distance_from_previous"
        ] = distance

        features[
            "time_since_previous_minutes"
        ] = time_minutes

        if time_seconds > 0:
            features["geo_velocity_kmh"] = (
                distance
                / (time_seconds / 3600)
            )
        else:
            features[
                "geo_velocity_kmh"
            ] = 0.0

    else:

        features[
            "distance_from_previous"
        ] = 0.0

        features[
            "time_since_previous_minutes"
        ] = -1.0

        features[
            "geo_velocity_kmh"
        ] = 0.0

    return features