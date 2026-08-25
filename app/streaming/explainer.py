import pandas as pd
import shap


class FraudExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(
            self.model
        )

    def explain(self, features, top_n=5):
        feature_names = (
            self.model.get_booster()
            .feature_names
        )

        missing = [
            name for name in feature_names if name not in features
        ]
        if missing:
            raise ValueError(
                f"Missing required feature(s) for explanation: {missing}"
            )

        X = pd.DataFrame(
            [[features[name] for name in feature_names]],
            columns=feature_names,
        )
        shap_values = self.explainer.shap_values(X)

        # SHAP's output shape for binary classifiers depends on the
        # version/explainer path:
        #  - list  -> [values_for_class_0, values_for_class_1], each
        #             shaped (n_samples, n_features). We want the
        #             positive (fraud) class, index 1, then the single
        #             sample's row.
        #  - array -> a single (n_samples, n_features) array already
        #             w.r.t. the model's single output. Just take the
        #             sample's row.
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        if len(values) != len(feature_names):
            raise ValueError(
                f"SHAP returned {len(values)} values for "
                f"{len(feature_names)} features — explainer output "
                f"shape doesn't match what this code expects, check "
                f"the installed shap version."
            )

        explanations = []
        for name, value, shap_value in zip(
            feature_names,
            X.iloc[0].values,
            values,
        ):
            explanations.append(
                {
                    "feature": name,
                    "feature_value": float(value),
                    "shap_value": float(shap_value),
                }
            )

        explanations.sort(
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )

        return explanations[:top_n]