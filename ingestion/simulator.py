from __future__ import annotations
import random
import uuid
from datetime import datetime, timezone
from typing import Any
from db.schema import STAGES, STATUS_NORMAL, STATUS_WARNING, STATUS_ANOMALY

_STAGE_CFG: dict[str, dict] = {
    "SUPPLIER":      {"temp": (1,   5),  "thresh": 8,   "delay": (0,  5),  "inv": (800, 1200)},
    "MANUFACTURING": {"temp": (2,   7),  "thresh": 10,  "delay": (0,  8),  "inv": (500,  900)},
    "WAREHOUSE":     {"temp": (1,   4),  "thresh": 7,   "delay": (0,  6),  "inv": (300,  700)},
    "TRANSPORT":     {"temp": (3,   8),  "thresh": 12,  "delay": (0, 15),  "inv": (50,   200)},
    "DELIVERY":      {"temp": (4,  10),  "thresh": 15,  "delay": (0, 20),  "inv": (10,    80)},
}

_TEMP_SPIKE_ADD   = {
    "SUPPLIER":      12,
    "MANUFACTURING": 15,
    "WAREHOUSE":     14,
    "TRANSPORT":     18,
    "DELIVERY":      16,
}
_DELAY_SPIKE_MUL  = 5.0
_INV_DROP_FRAC    = 0.20

_anomaly_intensity: dict[str, int] = {stage: 0 for stage in STAGES}


def get_anomaly_intensity(stage: str) -> int:
    return _anomaly_intensity.get(stage, 0)


def set_anomaly_intensity(stage: str, value: int) -> None:
    _anomaly_intensity[stage] = max(0, min(5, value))


def decay_anomaly_intensity(decay_by: int = 1) -> None:
    """Gradually reduce anomaly intensity across all stages."""
    for stage in STAGES:
        _anomaly_intensity[stage] = max(0, _anomaly_intensity[stage] - decay_by)


def _generate_event(
    stage: str,
    force_anomaly: bool = False,
    anomaly_rate: float = 0.02,
    drift_factor: float = 0.0,
) -> dict[str, Any]:
    cfg = _STAGE_CFG[stage]
    t_lo, t_hi = cfg["temp"]
    d_lo, d_hi = cfg["delay"]
    i_lo, i_hi = cfg["inv"]
    thresh      = cfg["thresh"]

    intensity = _anomaly_intensity[stage] if force_anomaly else 0
    is_anomaly = force_anomaly or (random.random() < anomaly_rate)

    if is_anomaly:
        anomaly_type = random.choice(["temp", "delay", "inventory", "combined"])
        base_spike = _TEMP_SPIKE_ADD[stage]
        extra_degrees = intensity * 2.5
        
        if anomaly_type in ("temp", "combined"):
            temp = round(thresh + random.uniform(2, base_spike + extra_degrees), 2)
        else:
            temp = round(random.uniform(t_lo, t_hi), 2)

        multiplier = _DELAY_SPIKE_MUL + (intensity * 2)
        if anomaly_type in ("delay", "combined"):
            delay = int(random.uniform(d_lo, d_hi) * multiplier + random.uniform(5, 20))
        else:
            delay = int(random.uniform(d_lo, d_hi))

        if anomaly_type == "inventory":
            drop_frac = max(0.05, _INV_DROP_FRAC - intensity * 0.03)
            inventory = int(i_lo * drop_frac)
        else:
            inventory = int(random.uniform(i_lo, i_hi))

        status = STATUS_ANOMALY if anomaly_type == "combined" else STATUS_WARNING
    else:
        temp_ceiling = t_lo + (thresh - t_lo) * drift_factor
        temp      = round(random.uniform(t_lo, min(t_hi + drift_factor * 3, temp_ceiling or t_hi)), 2)
        delay     = int(random.uniform(d_lo, d_hi + drift_factor * 5))
        inventory = int(random.uniform(i_lo, i_hi))
        status    = STATUS_NORMAL if drift_factor < 0.5 else STATUS_WARNING

    return {
        "timestamp":   datetime.now(timezone.utc),
        "entity_id":   f"{stage[:3]}-{uuid.uuid4().hex[:6].upper()}",
        "stage":       stage,
        "temperature": temp,
        "delay":       delay,
        "inventory":   inventory,
        "status":      status,
    }


def generate_batch(
    anomaly_rate: float = 0.15,
    force_anomaly_stage: str | None = None,
) -> list[dict[str, Any]]:
    """Generate one event per stage with propagation cascades."""
    events: list[dict[str, Any]] = []
    transport_anomaly = False

    for i, stage in enumerate(STAGES):
        force        = (stage == force_anomaly_stage)
        stage_rate   = anomaly_rate
        drift        = 0.0

        if i > 0:
            prev_evt = events[i-1]
            if prev_evt["status"] != STATUS_NORMAL:
                drift = 0.3 if prev_evt["status"] == STATUS_WARNING else 0.5
                stage_rate = min(0.6, anomaly_rate * 2)

        if stage == "DELIVERY" and transport_anomaly:
            stage_rate = min(0.8, anomaly_rate * 4)
            drift      = 0.7

        evt = _generate_event(stage, force_anomaly=force, anomaly_rate=stage_rate, drift_factor=drift)
        events.append(evt)

        if stage == "TRANSPORT" and evt["status"] != STATUS_NORMAL:
            transport_anomaly = True

    return events


def generate_training_data(n_per_stage: int = 400) -> list[dict[str, Any]]:
    """Synthetic data mix for initial ML training."""
    data: list[dict[str, Any]] = []
    for stage in STAGES:
        for i in range(n_per_stage):
            rate = 0.0 if i < int(n_per_stage * 0.95) else 1.0
            data.append(_generate_event(stage, force_anomaly=False, anomaly_rate=rate))
    return data