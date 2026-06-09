"""
my_model.py
-----------
Loads model.pkl (RandomForest + StandardScaler bundle) and exposes
predict_health() which returns a structured prediction used as the
Remarks field in the app.

Run model_training.py first to generate model.pkl.
"""

import pickle
import numpy as np

# ── Load bundle ───────────────────────────────────────────────────────────────
try:
    with open("model.pkl", "rb") as f:
        _bundle = pickle.load(f)
    _model  = _bundle["model"]
    _scaler = _bundle["scaler"]
except FileNotFoundError:
    raise RuntimeError(
        "model.pkl not found. Run 'python model_training.py' first to generate it."
    )
except Exception as e:
    raise RuntimeError(f"Failed to load model.pkl: {e}")


def predict_health(glucose: float, hemoglobin: float, cholesterol: float) -> dict:
    """
    Predicts health risk and returns a structured result.

    Args:
        glucose     : Fasting glucose in mg/dL
        hemoglobin  : Haemoglobin in g/dL
        cholesterol : Total cholesterol in mg/dL

    Returns:
        dict with keys:
            label      : "High Risk" or "Low Risk"
            confidence : float, e.g. 87.3
            reason     : short string explaining main driver
            remarks    : full formatted string for the Remarks column
    """
    # Input validation
    for name, val in [("glucose", glucose), ("hemoglobin", hemoglobin), ("cholesterol", cholesterol)]:
        if not isinstance(val, (int, float)):
            raise ValueError(f"{name} must be a number.")
        if val <= 0:
            raise ValueError(f"{name} must be a positive number.")

    # Scale and predict
    X          = _scaler.transform([[glucose, hemoglobin, cholesterol]])
    label_int  = _model.predict(X)[0]
    proba      = _model.predict_proba(X)[0]
    confidence = round(float(proba[label_int]) * 100, 1)
    label      = "High Risk" if label_int == 1 else "Low Risk"

    # Determine primary reason based on which value is most abnormal
    reasons = []
    if glucose > 125:
        reasons.append("elevated glucose (diabetic range)")
    elif glucose > 99:
        reasons.append("borderline glucose (pre-diabetic range)")

    if hemoglobin < 12:
        reasons.append("low haemoglobin (anaemia)")
    elif hemoglobin > 17.5:
        reasons.append("high haemoglobin")

    if cholesterol >= 240:
        reasons.append("high cholesterol")
    elif cholesterol >= 200:
        reasons.append("borderline cholesterol")

    if reasons:
        reason = ", ".join(reasons)
    elif label == "Low Risk":
        reason = "all values within normal range"
    else:
        reason = "combination of borderline values"

    remarks = f"{label} ({confidence}% confidence) — {reason}"

    return {
        "label":      label,
        "confidence": confidence,
        "reason":     reason,
        "remarks":    remarks,
    }
