from __future__ import annotations
import numpy as np
from features.feature_engine import extract_features
from ml import anomaly_model as model
from db.schema import STAGES

def compute_stage_risk(stage: str, events: list[dict]) -> dict:
    """
    Given a list of recent events for one stage, return a risk dict.
    """
    features = extract_features(events)

    if features[0] < 0:
        return {
            "stage":        stage,
            "risk_score":   0.0,
            "is_anomaly":   False,
            "ml_score":     0.0,
            "features":     {},
            "event_count":  0,
            "latest_event": None,
        }

    prediction = model.predict(features)

    feat_names = [
        "mean_temp", "max_temp", "std_temp",
        "mean_delay", "max_delay",
        "inventory_drop_rate", "anomaly_flag_ratio",
    ]

    ml_score = prediction["score"]
    rule_penalty = 0.0
    if features[6] > 0.3:
        rule_penalty += 15.0

    risk_score = min(100.0, round(ml_score + rule_penalty, 1))

    return {
        "stage":        stage,
        "risk_score":   risk_score,
        "is_anomaly":   prediction["is_anomaly"],
        "ml_score":     ml_score,
        "features":     dict(zip(feat_names, [round(float(f), 3) for f in features])),
        "event_count":  len(events),
        "latest_event": events[-1] if events else None,
    }


def compute_all_stage_risks(stage_events: dict[str, list[dict]]) -> dict[str, dict]:
    return {
        stage: compute_stage_risk(stage, stage_events.get(stage, []))
        for stage in STAGES
    }
