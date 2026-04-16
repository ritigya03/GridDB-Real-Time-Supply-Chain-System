from __future__ import annotations
import logging
import numpy as np
from ingestion.simulator import generate_training_data
from features.feature_engine import extract_features
from ml import anomaly_model as model

logger = logging.getLogger(__name__)


def train_if_needed() -> None:
    if model.is_ready():
        return

    logger.info("Training Isolation Forest on synthetic data...")
    raw_events  = generate_training_data(n_per_stage=300)
    feature_vecs: list[np.ndarray] = []

    window = 20
    for i in range(0, len(raw_events) - window, window // 2):
        chunk = raw_events[i : i + window]
        feature_vecs.append(extract_features(chunk))

    if not feature_vecs:
        feature_vecs = [extract_features(raw_events)]

    X = np.array(feature_vecs)
    model.train(X)
    logger.info("Bootstrap training complete.")
