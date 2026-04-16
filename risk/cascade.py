from __future__ import annotations
from db.schema import STAGES

# Downstream amplification weights
_CASCADE_WEIGHTS: dict[str, float] = {
    "SUPPLIER":      0.10,
    "MANUFACTURING": 0.15,
    "WAREHOUSE":     0.20,
    "TRANSPORT":     0.25,
    "DELIVERY":      0.30,
}

# Upstream leakage factor
_PROPAGATION_ALPHA = 0.15


def propagate_risk(stage_risks: dict[str, dict]) -> dict:
    """
    Compute global cascaded risk from individual stage risk data.
    """
    raw: dict[str, float] = {s: stage_risks[s].get("risk_score", 0.0) for s in STAGES if s in stage_risks}

    # Forward propagation: each stage inherits previous stage risk leakage
    propagated: dict[str, float] = {}
    prev_risk = 0.0
    for stage in STAGES:
        base  = raw.get(stage, 0.0)
        leaked = _PROPAGATION_ALPHA * prev_risk
        combined = min(100.0, base + leaked * (1 - base / 100.0))
        propagated[stage] = round(combined, 1)
        prev_risk = combined

    global_risk = round(sum(propagated[s] * _CASCADE_WEIGHTS[s] for s in STAGES), 1)
    highest = max(propagated, key=propagated.get)

    if global_risk < 30:
        level = "LOW"
    elif global_risk < 55:
        level = "MEDIUM"
    elif global_risk < 75:
        level = "HIGH"
    else:
        level = "CRITICAL"

    return {
        "global_risk":        global_risk,
        "risk_level":         level,
        "highest_risk_stage": highest,
        "stage_cascade":      propagated,
        "cascade_path":       [(s, propagated[s]) for s in STAGES],
    }
