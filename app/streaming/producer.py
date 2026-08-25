import json
import uuid

from kafka import KafkaProducer

from app.streaming.events import TransactionEvent


BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "transactions"


producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    key_serializer=lambda key: key.encode("utf-8"),
    acks="all",
    enable_idempotence=True,
    retries=5,
)


def publish_transaction(event: TransactionEvent):
    # Keyed by account_id: all transactions for the same account land on
    # the same partition, which is what guarantees the consumer sees
    # them in order. Velocity/geo-based fraud features (e.g. the
    # impossible-travel case below) depend on that ordering — without a
    # key, the default partitioner can spread same-account transactions
    # across partitions with no ordering guarantee between them.
    future = producer.send(
        TOPIC,
        key=event.account_id,
        value=event.model_dump(),
    )

    metadata = future.get(timeout=10)

    print(
        f"Published {event.transaction_id} "
        f"to {metadata.topic}:{metadata.partition}"
        f"@{metadata.offset}"
    )


if __name__ == "__main__":

    run_id = uuid.uuid4().hex[:8]

    events = [
        TransactionEvent(
            transaction_id=f"TXN_LIVE_{run_id}_000001",
            account_id="ACC_000001",
            amount=850.0,
            merchant="Amazon",
            merchant_base_risk=0.01,
            latitude=28.6139,
            longitude=77.2090,
            timestamp="2026-08-25T04:00:00+00:00",
        ),
        TransactionEvent(
            transaction_id=f"TXN_LIVE_{run_id}_000002",
            account_id="ACC_000001",
            amount=950.0,
            merchant="Amazon",
            merchant_base_risk=0.01,
            latitude=19.0760,
            longitude=72.8777,
            timestamp="2026-08-25T04:05:00+00:00",
        ),
    ]

    for event in events:
        publish_transaction(event)

    producer.flush()
    producer.close()