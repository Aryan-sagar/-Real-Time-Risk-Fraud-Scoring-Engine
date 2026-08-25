from app.streaming.feature_state import FeatureState


def test_redis_state():

    state_manager = FeatureState()

    state_manager.redis.flushdb()

    state_manager.add_transaction(
        account_id="ACC_TEST",
        timestamp="2026-08-25T03:00:00",
        amount=500.0,
        latitude=28.6139,
        longitude=77.2090,
    )

    state = state_manager.get_account_state(
        "ACC_TEST"
    )

    assert state["transaction_count"] == 1

    assert state["total_amount"] == 500.0

    assert len(state["transactions"]) == 1