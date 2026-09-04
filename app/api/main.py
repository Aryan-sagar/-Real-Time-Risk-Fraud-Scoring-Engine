from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.streaming.feature_state import FeatureState
from app.streaming.feature_assembler import FeatureAssembler
from app.streaming.model_scorer import FraudModelScorer
from app.streaming.explainer import FraudExplainer


app = FastAPI(
    title="Fintech Risk Engine",
    version="1.0.0",
)


feature_state = FeatureState()
assembler = FeatureAssembler(feature_state)
scorer = FraudModelScorer()
explainer = FraudExplainer(scorer.model)


class ScoreRequest(BaseModel):
    transaction_id: str
    account_id: str
    amount: float = Field(gt=0)
    merchant: str
    merchant_base_risk: float
    latitude: float
    longitude: float
    timestamp: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "fintech-risk-engine",
    }


@app.post("/score")
def score_transaction(request: ScoreRequest):

    # Idempotency check — must happen before feature building/scoring,
    # not just before persistence, or a duplicate still gets scored
    # twice (wasted SHAP/inference work) even if state doesn't drift.
    if not feature_state.mark_transaction_processed(request.transaction_id):
        return {
            "transaction_id": request.transaction_id,
            "decision": "DUPLICATE_IGNORED",
        }

    transaction = {
        "transaction_id": request.transaction_id,
        "account_id": request.account_id,
        "amount": request.amount,
        "merchant": request.merchant,
        "merchant_base_risk": (
            request.merchant_base_risk
        ),
        "latitude": request.latitude,
        "longitude": request.longitude,
        "timestamp": request.timestamp,
    }

    # Get account history BEFORE current transaction.
    previous_transactions = (
        feature_state.get_recent_transactions(
            request.account_id,
            request.timestamp,
        )
    )

    # Build exact model feature vector.
    features = assembler.build(
        transaction,
        previous_transactions,
    )

    # Score transaction — uses scorer.threshold (calibrated at training
    # time), not a hardcoded value. Keeps this path consistent with the
    # Kafka consumer's decision logic.
    result = scorer.predict(features)

    # Explain only when useful.
    explanations = []

    if result["decision"] == "BLOCK":
        explanations = explainer.explain(
            features,
            top_n=5,
        )

    # Persist transaction state.
    state = feature_state.add_transaction(
        transaction_id=request.transaction_id,
        account_id=request.account_id,
        timestamp=request.timestamp,
        amount=request.amount,
        latitude=request.latitude,
        longitude=request.longitude,
        merchant=request.merchant,
        merchant_base_risk=(
            request.merchant_base_risk
        ),
    )

    return {
        "transaction_id": request.transaction_id,
        "risk_probability": result[
            "risk_probability"
        ],
        "decision": result["decision"],
        "features": features,
        "explanations": explanations,
        "state": {
            "recent_transaction_count": (
                state["transaction_count"]
                if state is not None
                else None
            ),
        },
    }