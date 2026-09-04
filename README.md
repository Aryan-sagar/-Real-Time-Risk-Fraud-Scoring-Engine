# Real-Time Risk & Fraud Scoring Engine

A production-style real-time fraud detection system for fintech transactions.

The system consumes transaction events, builds stateful behavioral features using Redis, scores transactions with a calibrated XGBoost model, generates SHAP explanations for blocked transactions, and routes malformed events to a Kafka Dead Letter Queue (DLQ).

---

## Architecture

```text
                         ┌─────────────────────┐
                         │  Transaction Events  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Redpanda / Kafka    │
                         │    transactions     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │      Risk Engine Consumer    │
                    │                              │
                    │  • Validation                │
                    │  • Idempotency               │
                    │  • Feature construction      │
                    │  • Model inference           │
                    └──────────────┬───────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
             ┌────────────────┐          ┌────────────────┐
             │ Redis Feature  │          │ XGBoost Model  │
             │     State      │          │                │
             └────────────────┘          └───────┬────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │ Risk Probability│
                                        │ + Decision      │
                                        └────────┬────────┘
                                                 │
                                    ┌────────────┴────────────┐
                                    ▼                         ▼
                                 ALLOW                     BLOCK
                                                              │
                                                              ▼
                                                        SHAP Explanation


Malformed Events
       │
       ▼
transactions.dlq
````

---

## Features

### Stateful Transaction Features

The engine maintains per-account transaction state in Redis and derives real-time behavioral signals including:

* Transaction count in 1 minute
* Transaction count in 10 minutes
* Transaction count in 1 hour
* Amount sum in 10 minutes
* Amount sum in 1 hour
* Account average transaction amount
* Account transaction standard deviation
* Amount deviation from account behavior
* Distance from previous transaction
* Time since previous transaction
* Geo-velocity
* Merchant base risk
* Merchant fraud rate
* Hour of day
* Day of week
* Weekend indicator

These features allow the model to detect behavioral anomalies rather than relying only on the current transaction.

---

## Machine Learning

The fraud classifier uses **XGBoost**.

The model was trained on 150,300 synthetic fintech transactions containing 2,193 fraudulent transactions.

### Dataset

| Metric                  |   Value |
| ----------------------- | ------: |
| Transactions            | 150,300 |
| Fraudulent transactions |   2,193 |
| Fraud rate              |   1.46% |

### Model Performance

#### Validation

| Metric    |  Score |
| --------- | -----: |
| PR-AUC    | 0.8206 |
| ROC-AUC   | 0.9961 |
| Precision | 0.6994 |
| Recall    | 0.7423 |
| F1        | 0.7202 |

#### Test

| Metric    |  Score |
| --------- | -----: |
| PR-AUC    | 0.7192 |
| ROC-AUC   | 0.9945 |
| Precision | 0.6538 |
| Recall    | 0.6608 |
| F1        | 0.6573 |

Because fraud detection is highly imbalanced, PR-AUC is particularly important when evaluating the model.

---

## Calibrated Decision Threshold

Instead of using the default `0.5` classification threshold, the system selects a threshold using the validation set.

Current calibrated threshold:

```text
0.9168
```

This threshold is persisted in:

```text
data/processed/fraud_xgboost.metadata.json
```

Both the FastAPI scoring path and Kafka consumer use the same calibrated threshold.

This keeps online decisions consistent with the model evaluation process.

---

## Explainable Fraud Decisions

Blocked transactions are explained using **SHAP**.

For example, a simulated suspicious transaction produced:

```text
Risk probability: 0.9900
Decision: BLOCK
```

Strong positive contributors included:

```text
geo_velocity_kmh     +3.10
txn_count_10m        +2.33
```

The transaction involved:

```text
Amount:                    $950
Merchant risk:             0.80
Distance from previous:    1354 km
Time since previous:       5 minutes
Geo velocity:              16,251 km/h
```

This allows the system to answer not only:

> "Is this transaction risky?"

but also:

> "Why did the system block it?"

---

## Real-Time Idempotency

The engine prevents duplicate transaction processing using Redis.

The implementation uses Redis `SADD` as an atomic check-and-mark operation:

```text
SADD processed_transactions <transaction_id>
```

Redis returns:

```text
1 → new transaction
0 → duplicate
```

This avoids a race condition between separate `SISMEMBER` and `SADD` operations.

Example:

```text
First request
    ↓
BLOCK

Same transaction again
    ↓
DUPLICATE_IGNORED
```

---

## Dead Letter Queue

Malformed Kafka events are not allowed to crash the consumer.

Invalid events are routed to:

```text
transactions.dlq
```

Example test:

```json
{
  "transaction_id": "DLQ_TEST_001",
  "account_id": "ACC_BAD_001"
}
```

The event was successfully captured in the DLQ while the consumer continued running.

---

## REST API

FastAPI exposes a real-time scoring endpoint.

### Health Check

```http
GET /health
```

Example:

```json
{
  "status": "healthy"
}
```

### Score Transaction

```http
POST /score
```

Example request:

```json
{
  "transaction_id": "TXN_001",
  "account_id": "ACC_001",
  "amount": 950,
  "merchant": "Suspicious Merchant",
  "merchant_base_risk": 0.8,
  "latitude": 19.076,
  "longitude": 72.8777,
  "timestamp": "2026-09-05T05:05:00"
}
```

Example response:

```json
{
  "transaction_id": "TXN_001",
  "risk_probability": 0.9900,
  "decision": "BLOCK"
}
```

---

## Technology Stack

| Component      | Technology       |
| -------------- | ---------------- |
| Language       | Python           |
| API            | FastAPI          |
| ML             | XGBoost          |
| Explainability | SHAP             |
| Streaming      | Redpanda / Kafka |
| State Store    | Redis            |
| Validation     | Pydantic         |
| Testing        | Pytest           |
| Containers     | Docker           |
| Orchestration  | Docker Compose   |

---

## Project Structure

```text
real-time-risk-fraud-scoring-engine/
│
├── app/
│   ├── api/
│   │   └── main.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── scorer.py
│   │   └── merchant_risk.py
│   │
│   ├── streaming/
│   │   ├── consumer.py
│   │   ├── events.py
│   │   └── feature_state.py
│   │
│   └── explainability/
│       └── ...
│
├── data/
│   ├── raw/
│   ├── synthetic/
│   └── processed/
│
├── scripts/
│   ├── generate_transactions.py
│   └── inject_realistic_fraud.py
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Running Locally

### 1. Clone

```bash
git clone https://github.com/Aryan-sagar/Fintech-risk-engine.git
cd Fintech-risk-engine
```

### 2. Create virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Start infrastructure

```powershell
docker compose up -d redis redpanda
```

Verify Redis:

```powershell
docker exec fintech-risk-redis redis-cli ping
```

Expected:

```text
PONG
```

### 5. Start the API

```powershell
uvicorn app.api.main:app --reload
```

API:

```text
http://localhost:8000
```

### 6. Start the Kafka consumer

From the project root:

```powershell
python -m app.streaming.consumer
```

The consumer listens to:

```text
transactions
```

and sends invalid events to:

```text
transactions.dlq
```

---

## Docker

Run the complete stack:

```powershell
docker compose up -d --build
```

Check services:

```powershell
docker compose ps
```

Check risk engine logs:

```powershell
docker logs fintech-risk-engine
```

---

## Testing

Run the test suite:

```powershell
python -m pytest -q
```

Current automated test status:

```text
2 passed
```

The system has additionally been manually tested for:

* REST transaction scoring
* Normal transaction → `ALLOW`
* Suspicious transaction → `BLOCK`
* Duplicate transaction → `DUPLICATE_IGNORED`
* Kafka normal event → `ALLOW`
* Kafka fraud event → `BLOCK`
* Malformed Kafka event → DLQ
* Redis state updates
* SHAP fraud explanations

---

## Example Fraud Scenario

A legitimate transaction:

```text
Amount: $60
Merchant risk: 0.05
```

was allowed:

```text
risk_probability ≈ 0.00018
decision = ALLOW
```

A second transaction from the same account shortly afterward:

```text
Amount: $950
Merchant risk: 0.80
Distance: 1354 km
Time difference: 5 minutes
Geo velocity: 16,251 km/h
```

was blocked:

```text
risk_probability ≈ 0.99004
decision = BLOCK
```

The system also generated SHAP explanations identifying the strongest contributing features.

---

## Engineering Highlights

This project demonstrates more than model training. It focuses on the engineering required to serve ML models in a real-time environment:

* Stateful online feature engineering
* Streaming event processing
* Redis-backed feature state
* Atomic transaction idempotency
* Calibrated model thresholds
* Explainable fraud decisions
* Kafka/Redpanda consumer groups
* Dead Letter Queue handling
* REST inference API
* Dockerized infrastructure
* Automated testing
* Separation of training and inference artifacts

---

## Future Improvements

Planned improvements include:

* [ ] Run Kafka consumer as a dedicated Docker Compose service
* [ ] Improve idempotency lifecycle with processing/success states
* [ ] Add comprehensive integration tests
* [ ] Add Prometheus metrics
* [ ] Add Grafana monitoring dashboard
* [ ] Add model drift monitoring
* [ ] Add structured JSON logging
* [ ] Add CI/CD with GitHub Actions
* [ ] Add load testing for the scoring API
* [ ] Add model versioning and registry support

---

## Author

**Aryan Sagar**

B.Tech, IIT Ropar

Interested in Data Engineering, Machine Learning, AI Systems, and FinTech infrastructure.

```


