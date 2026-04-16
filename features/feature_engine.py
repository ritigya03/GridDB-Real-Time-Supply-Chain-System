from __future__ import annotations
import numpy as np
from typing import Any


def extract_features(events: list[dict[str, Any]]) -> np.ndarray:
    """
    Extract a 7-element feature vector from a list of raw events.
    [mean_temp, max_temp, std_temp, mean_delay, max_delay, inventory_drop, anomaly_ratio]
    """
    if not events:
        return np.full(7, -1.0)

    temps     = [float(e.get("temperature", 0)) for e in events]
    delays    = [float(e.get("delay",       0)) for e in events]
    inventories = [float(e.get("inventory", 0)) for e in events]
    statuses  = [e.get("status", "NORMAL") for e in events]

    n = len(events)
    mean_temp  = float(np.mean(temps))
    max_temp   = float(np.max(temps))
    std_temp   = float(np.std(temps)) if n > 1 else 0.0
    mean_delay = float(np.mean(delays))
    max_delay  = float(np.max(delays))

    inv_range  = max(inventories) - min(inventories) if n > 1 else 0.0
    inventory_drop_rate = inv_range / n
    anomaly_flag_ratio = sum(1 for s in statuses if s != "NORMAL") / n

    return np.array([
        mean_temp, max_temp, std_temp,
        mean_delay, max_delay,
        inventory_drop_rate,
        anomaly_flag_ratio
    ], dtype=float)


def extract_all_stages(stage_events: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    return {stage: extract_features(evts) for stage, evts in stage_events.items()}


def feature_names() -> list[str]:
    return [
        "mean_temp", "max_temp", "std_temp",
        "mean_delay", "max_delay",
        "inventory_drop_rate", "anomaly_flag_ratio"
    ]
