"""Training-only fitting and validation-only hyperparameter selection."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import RANDOM_STATE
from evaluate import classification_metrics


def train_and_select(x_train, y_train, x_val, y_val):
    scaler = StandardScaler()
    scaler.fit(x_train)
    train_scaled = scaler.transform(x_train)
    val_scaled = scaler.transform(x_val)
    records, best_model, best_f1 = [], None, -1.0
    for kernel in ("linear", "rbf"):
        for c in (0.1, 1, 10, 100):
            for gamma in ((None,) if kernel == "linear" else ("scale", 0.01, 0.1, 1)):
                params = {"kernel": kernel, "C": c, "random_state": RANDOM_STATE}
                if gamma is not None:
                    params["gamma"] = gamma
                print(f"Training candidate: {params}", flush=True)
                model = SVC(**params)
                model.fit(train_scaled, y_train)
                if not np.array_equal(model.classes_, [0, 1]):
                    raise ValueError("Expected SVC classes [0, 1]; positive scores must mean spoof")
                metrics = classification_metrics(y_val, model.predict(val_scaled),
                                                 model.decision_function(val_scaled))
                records.append({"kernel": kernel, "C": c, "gamma": gamma,
                                **{f"validation_{key}": value for key, value in metrics.items()}})
                print(records[-1], flush=True)
                # Deterministic ties: keep first candidate in the documented grid order.
                if metrics["f1"] > best_f1:
                    best_model, best_f1 = model, metrics["f1"]
    table = pd.DataFrame(records)
    best = table.loc[table.validation_f1.idxmax()].to_dict()
    if best["gamma"] is None or pd.isna(best["gamma"]):
        best["gamma"] = None
    print(f"Best model (validation F1): {best}", flush=True)
    return best_model, scaler, table, best
