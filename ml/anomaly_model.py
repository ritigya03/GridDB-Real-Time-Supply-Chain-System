from __future__ import annotations
import logging
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model.pkl"
_model: IsolationForest | None = None


def _load_or_none() -> IsolationForest | None:
    if MODEL_PATH.exists():
        try:
            m = joblib.load(MODEL_PATH)
            logger.info("Loaded Isolation Forest from %s", MODEL_PATH)
            return m
        except Exception as exc:
            logger.warning("Could not load model: %s", exc)
    return None


def train(X: np.ndarray) -> None:
    """Fit and persist the Isolation Forest model."""
    global _model
    _model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        max_samples="auto",
    )
    _model.fit(X)
    joblib.dump(_model, MODEL_PATH)
    logger.info("Isolation Forest trained on %d samples.", len(X))


def predict(features: np.ndarray) -> dict:
    """Predict anomaly for a single feature vector."""
    global _model
    if _model is None:
        _model = _load_or_none()
        if _model is None:
            return {"is_anomaly": False, "score": 20.0, "raw_score": 0.0}

    vec = features.reshape(1, -1)
    prediction  = _model.predict(vec)[0]  # 1 = normal, -1 = anomaly
    raw_score   = float(_model.decision_function(vec)[0])

    # Map raw decision function scores to 0-100 scale.
    clamped = max(-0.4, min(0.4, raw_score))
    score   = round(100 * (0.4 - clamped) / 0.8, 1) 

    return {
        "is_anomaly": bool(prediction == -1),
        "score":      score,
        "raw_score":  raw_score,
    }


def is_ready() -> bool:
    global _model
    if _model is not None:
        return True
    _model = _load_or_none()
    return _model is not None
