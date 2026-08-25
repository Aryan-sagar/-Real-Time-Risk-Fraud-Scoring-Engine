import json
import os

import pandas as pd
import xgboost as xgb


MODEL_PATH = "data/processed/fraud_xgboost.json"
METADATA_PATH = "data/processed/fraud_xgboost.metadata.json"

# Fallback only, used if no metadata file is found next to the model.
DEFAULT_THRESHOLD = 0.5


class FraudModelScorer:
    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.feature_names = (
            self.model.get_booster().feature_names
        )

        self.threshold = DEFAULT_THRESHOLD
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.threshold = metadata.get("threshold", DEFAULT_THRESHOLD)
        else:
            # Falling back to an arbitrary default here is exactly the
            # problem the validation-derived threshold in training was
            # meant to fix — don't let that happen silently.
            print(
                f"WARNING: no metadata file at {metadata_path}; "
                f"using fallback threshold {DEFAULT_THRESHOLD}. "
                f"Re-run training to produce a calibrated threshold."
            )

    def predict_probability(self, features):
        missing = [
            name for name in self.feature_names if name not in features
        ]
        if missing:
            raise ValueError(
                f"Missing required feature(s) for scoring: {missing}"
            )

        # Force exact training feature order.
        X = pd.DataFrame(
            [[features[name] for name in self.feature_names]],
            columns=self.feature_names,
        )
        probability = self.model.predict_proba(
            X
        )[0][1]
        return float(probability)

    def predict(self, features, threshold=None):
        if threshold is None:
            threshold = self.threshold

        probability = self.predict_probability(
            features
        )
        if probability >= threshold:
            decision = "BLOCK"
        else:
            decision = "ALLOW"
        return {
            "risk_probability": probability,
            "decision": decision,
            "threshold": threshold,
        }