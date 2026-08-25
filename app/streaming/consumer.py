import json
import logging
import traceback

from kafka import KafkaConsumer, KafkaProducer

from app.streaming.events import TransactionEvent
from app.streaming.feature_state import FeatureState
from app.streaming.feature_assembler import FeatureAssembler
from app.streaming.model_scorer import FraudModelScorer
from app.streaming.explainer import FraudExplainer


BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "transactions"
DLQ_TOPIC = "transactions.dlq"
GROUP_ID = "risk-engine-debug-v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("risk_engine_consumer")


consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    group_id=GROUP_ID,
    auto_offset_reset="earliest",
    enable_auto_commit=False,
    value_deserializer=lambda value: json.loads(
        value.decode("utf-8")
    ),
)

# Separate producer used only to dead-letter messages we can't process.
# The point: a message is NEVER just skipped. It's either processed
# successfully, or it's durably written to the DLQ, *before* we ever
# advance the consumer offset past it. consumer.commit() with no args
# commits the current fetch position for all partitions — so if you let
# a failed message's offset get silently overtaken by the next
# successful commit, that transaction is gone with no record it ever
# existed. This closes that gap.
dlq_producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    acks="all",
)

feature_state = FeatureState()
assembler = FeatureAssembler(feature_state)
scorer = FraudModelScorer()
explainer = FraudExplainer(scorer.model)


def send_to_dlq(raw_value, error, stage):
    dlq_producer.send(
        DLQ_TOPIC,
        value={
            "stage": stage,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "raw_message": raw_value,
        },
    )
    dlq_producer.flush()


logger.info("Risk engine consumer started.")
logger.info(f"Listening to topic: {TOPIC}")


for message in consumer:

    raw_value = message.value

    # -----------------------------------------------------
    # Parse and validate the event
    # -----------------------------------------------------

    try:
        event = TransactionEvent(**raw_value)
    except Exception as exc:
        logger.error(f"Malformed event, sending to DLQ: {exc}")
        send_to_dlq(raw_value, exc, stage="parse")
        consumer.commit()
        continue

    logger.info(
        f"Received {event.transaction_id} "
        f"for {event.account_id} "
        f"at {event.timestamp} "
        f"(₹{event.amount})"
    )

    # -----------------------------------------------------
    # Idempotency
    # -----------------------------------------------------

    if feature_state.transaction_processed(event.transaction_id):
        logger.info(f"Duplicate ignored: {event.transaction_id}")
        consumer.commit()
        continue

    # -----------------------------------------------------
    # Build features and persist
    # -----------------------------------------------------

    try:
        # Previous state fetched BEFORE inserting the current event.
        previous_transactions = feature_state.get_recent_transactions(
            event.account_id,
            event.timestamp,
        )

        transaction = {
            "transaction_id": event.transaction_id,
            "account_id": event.account_id,
            "amount": event.amount,
            "merchant": event.merchant,
            "merchant_base_risk": event.merchant_base_risk,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "timestamp": event.timestamp,
        }

        logger.debug(
            f"DEBUG CURRENT: {transaction['transaction_id']} "
            f"{transaction['timestamp']}"
        )
        for tx in previous_transactions:
            logger.debug(
                f"DEBUG PREVIOUS: {tx['transaction_id']} "
                f"{tx['timestamp']} {tx['amount']}"
            )

        features = assembler.build(transaction, previous_transactions)

        logger.info("Model features:")
        for name, value in features.items():
            logger.info(f"  {name}: {value}")

        # No explicit threshold here on purpose — scorer.threshold is
        # the value calibrated from the validation PR curve at
        # training time. Hardcoding 0.5 here would silently undo that.
        result = scorer.predict(features)

        logger.info("🚨 RISK DECISION")
        logger.info(f"  Risk probability: {result['risk_probability']:.4f}")
        logger.info(
            f"  Decision: {result['decision']} "
            f"(threshold={result['threshold']:.4f})"
        )

        # Only explain blocked transactions — SHAP is real overhead per
        # message, and it's only actionable when there's a decision to
        # justify.
        if result["decision"] == "BLOCK":
            explanations = explainer.explain(
                features,
                top_n=5,
            )

            logger.info("🔎 WHY THIS TRANSACTION WAS BLOCKED")
            for explanation in explanations:
                direction = (
                    "increased risk"
                    if explanation["shap_value"] > 0
                    else "reduced risk"
                )
                logger.info(
                    f"  {explanation['feature']}: "
                    f"{explanation['feature_value']:.4f} "
                    f"→ SHAP "
                    f"{explanation['shap_value']:+.4f} "
                    f"({direction})"
                )

        # Persist AFTER feature calculation, as before.
        state = feature_state.add_transaction(
            transaction_id=event.transaction_id,
            account_id=event.account_id,
            timestamp=event.timestamp,
            amount=event.amount,
            latitude=event.latitude,
            longitude=event.longitude,
            merchant=event.merchant,
            merchant_base_risk=event.merchant_base_risk,
        )

        if state is None:
            # We already passed transaction_processed() above, so
            # add_transaction() disagreeing means a real inconsistency
            # (e.g. a race between the two checks) — not a normal
            # duplicate path. Worth surfacing, not silently passing.
            logger.warning(
                f"add_transaction() reported duplicate for "
                f"{event.transaction_id} despite passing the earlier "
                f"idempotency check — check FeatureState consistency."
            )
            consumer.commit()
            continue

        logger.info(
            f"Account {event.account_id}: "
            f"{state['transaction_count']} "
            f"recent transactions"
        )

        consumer.commit()

    except Exception as exc:
        # A failure here means we can't guarantee this transaction was
        # scored. DLQ it before committing so nothing is dropped
        # without a trace.
        logger.error(f"Failed to process {event.transaction_id}: {exc}")
        send_to_dlq(raw_value, exc, stage="feature_build_or_persist")
        consumer.commit()