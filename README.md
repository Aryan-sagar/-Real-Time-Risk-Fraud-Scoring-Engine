Absolutely bro. Paste this **directly into `README.md`**:

````markdown
# Fintech Real-Time Fraud Risk Engine

A production-style real-time fraud detection system designed to score financial transactions in milliseconds using streaming data, online behavioral features, XGBoost, Redis state, and SHAP explainability.

The system combines machine learning with real-time transaction state to detect suspicious behavior such as unusual transaction amounts, high transaction velocity, risky merchants, account takeover patterns, and impossible geographic movement.

---

## Architecture

```text
                    ┌──────────────────────┐
                    │      Transaction     │
                    │        Event          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Redpanda       │
                    │   Event Streaming     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Streaming Consumer │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
             ┌─────────────┐       ┌──────────────┐
             │    Redis    │       │    Feature   │
             │ Online State│──────►│   Assembler  │
             └─────────────┘       └──────┬───────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │    XGBoost    │
                                  │ Fraud Scoring │
                                  └───────┬───────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │     SHAP      │
                                  │ Explainability│
                                  └───────┬───────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │ ALLOW / BLOCK │
                                  └───────────────┘


              ┌────────────────────────────┐
              │          FastAPI           │
              │         POST /score        │
              └────────────┬───────────────┘
                           │
                           ▼
                    Same Risk Engine
````

---

## Key Features

### Real-Time Transaction Processing

Transactions can be consumed from a Redpanda/Kafka-compatible event stream and processed continuously.

### Online Behavioral Features

The system maintains recent account activity using Redis and calculates features such as:

* Transaction count over 1 minute
* Transaction count over 10 minutes
* Transaction count over 1 hour
* Amount spent over 10 minutes
* Amount spent over 1 hour
* Distance from previous transaction
* Time since previous transaction
* Geographic velocity
* Account average transaction amount
* Account transaction standard deviation
* Amount deviation from account behavior
* Merchant fraud rate
* Merchant base risk
* Hour of day
* Day of week
* Weekend indicator

### Machine Learning

An XGBoost classifier is used for fraud probability estimation.

The model was trained using a highly imbalanced fraud dataset and evaluated using PR-AUC and ROC-AUC rather than relying only on accuracy.

Validation performance:

* **PR-AUC: 0.8232**
* **ROC-AUC: 0.9962**

Test performance:

* **PR-AUC: 0.7165**
* **ROC-AUC: 0.9946**

The difference between validation and test performance highlights the importance of evaluating fraud models on unseen data rather than relying on a single metric.

### Threshold Analysis

The system supports configurable decision thresholds.

Example at threshold `0.50`:

* Precision: **0.4522**
* Recall: **0.8693**
* False positives: **298**
* False negatives: **37**

The threshold can be adjusted depending on the desired balance between fraud detection and customer friction.

---

## Fraud Types

The synthetic transaction generator injects multiple realistic fraud patterns:

* `account_takeover`
* `unusual_amount`
* `risky_merchant`
* `geo_velocity`
* `high_velocity`

Fraud distribution in the generated dataset:

```text
Total transactions: 150,300
Fraudulent transactions: 2,193
Fraud rate: 1.46%
```

---

## Explainable AI

The system uses SHAP to explain individual fraud decisions.

Example:

```text
Risk probability: 0.9997
Decision: BLOCK

Why?

geo_velocity_kmh       +4.7016
txn_count_10m          +2.9648
distance_from_previous +1.2279
amount_sum_10m         +0.7971

account_avg_amount     -1.0909
```

This allows the system to answer not only:

> "Is this transaction suspicious?"

but also:

> "Why was this transaction blocked?"

For example, a transaction can be flagged because an account apparently moved more than 1,100 km within five minutes while making several transactions within a short time window.

---

## Real-Time State Management

Redis maintains short-lived account state.

The state layer provides:

* Recent transaction history
* Rolling transaction windows
* Amount aggregation
* Geographic history
* Transaction idempotency
* Automatic state expiration

Transactions are protected against duplicate processing using Redis-backed idempotency.

---

## API

The system exposes a FastAPI endpoint:

```text
POST /score
```

Example request:

```json
{
  "transaction_id": "TXN_001",
  "account_id": "ACC_001",
  "amount": 950.0,
  "merchant": "Amazon",
  "merchant_base_risk": 0.01,
  "latitude": 19.0760,
  "longitude": 72.8777,
  "timestamp": "2026-08-25T04:05:00+00:00"
}
```

Example response:

```json
{
  "transaction_id": "TXN_001",
  "risk_probability": 0.9997,
  "decision": "BLOCK",
  "explanations": [
    {
      "feature": "geo_velocity_kmh",
      "shap_value": 4.7
    }
  ]
}
```

Health check:

```text
GET /health
```

---

## Docker

The complete application is containerized.

Services:

```text
risk-engine
redis
redpanda
```

Start the complete stack:

```bash
docker compose up -d
```

Check running services:

```bash
docker compose ps
```

API:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

---

## Testing

The project includes automated tests for the state-management and API layers.

Run:

```bash
python -m pytest tests -v
```

The test suite verifies functionality including:

* Redis-backed feature state
* Transaction storage
* Rolling transaction history
* Idempotent transaction processing
* API health endpoint
* Transaction scoring endpoint
* Valid risk decisions

---

## Project Structure

```text
fintech-risk-engine/
│
├── app/
│   ├── api/
│   ├── features/
│   ├── models/
│   └── streaming/
│
├── data/
│   ├── processed/
│   └── synthetic/
│
├── notebooks/
│   └── 01_eda.ipynb
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

Generated CSV datasets are intentionally excluded from Git tracking. They can be regenerated using the project scripts.

---

## Tech Stack

| Component        | Technology       |
| ---------------- | ---------------- |
| Language         | Python           |
| Machine Learning | XGBoost          |
| Explainability   | SHAP             |
| Data Processing  | Pandas, NumPy    |
| API              | FastAPI          |
| Streaming        | Redpanda / Kafka |
| Online State     | Redis            |
| Containerization | Docker           |
| Testing          | Pytest           |
| Model Evaluation | PR-AUC, ROC-AUC  |

---

## Design Principles

### Train/Serve Feature Parity

The same feature definitions are maintained between offline model training and online inference to reduce training-serving skew.

### Stateful Real-Time Inference

Fraud decisions depend not only on the current transaction but also on recent account behavior.

### Idempotent Processing

Duplicate transaction events are detected and ignored using Redis-backed transaction tracking.

### Explainability

High-risk decisions are accompanied by SHAP-based feature contributions.

### Configurable Risk Threshold

The decision boundary can be adjusted depending on the operational cost of false positives versus false negatives.

---

## Example Detection Scenario

Consider an account making:

```text
04:00 — ₹850 — Delhi
04:05 — ₹950 — Mumbai
```

The system observes:

```text
Distance: ~1,148 km
Time difference: 5 minutes
Geographic velocity: ~13,777 km/h
```

The transaction is therefore highly suspicious.

The model can produce:

```text
Risk probability: 99%+
Decision: BLOCK
```

with geographic velocity, transaction velocity, and distance contributing strongly to the decision.

---

## Limitations

This project uses a synthetic transaction dataset and therefore should not be considered a production fraud model without further validation.

A production deployment would require:

* Real transaction data
* Model calibration
* Temporal cross-validation
* Drift monitoring
* Feature freshness monitoring
* Model retraining pipelines
* Alert investigation workflows
* Production observability
* Authentication and authorization
* Rate limiting
* Secure secret management
* High-availability infrastructure

---

## Future Improvements

Potential extensions include:

* Kafka/Redpanda multi-partition scaling
* Feature store integration
* Model monitoring
* Data drift detection
* Probability calibration
* Cost-sensitive threshold optimization
* Online model retraining
* Grafana/Prometheus observability
* Fraud analyst dashboard
* Model registry
* Cloud deployment
* CI/CD pipeline
* Kubernetes deployment

---

## Project Goal

The goal of this project was to build more than a static fraud-classification notebook.

It demonstrates how a machine learning model can be integrated into a **stateful, explainable, real-time transaction risk system** capable of consuming events, generating behavioral features, scoring transactions, explaining decisions, and returning an operational `ALLOW` or `BLOCK` decision.

**Built as an end-to-end fintech ML engineering project.**

```
```
