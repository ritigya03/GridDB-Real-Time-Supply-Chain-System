from __future__ import annotations
import asyncio
import logging
import os
from collections import deque
from datetime import datetime, timezone
from dotenv import load_dotenv
from db import griddb_client as db
from ingestion.simulator import generate_batch, set_anomaly_intensity, get_anomaly_intensity, decay_anomaly_intensity

_STAGE_INFO = {
    "SUPPLIER":      {"company": "Source Dairy Facility",      "city": "Anand"},
    "MANUFACTURING": {"company": "Regional Processing Center", "city": "Delhi"},
    "WAREHOUSE":     {"company": "Centralized Cold Storage",   "city": "Jaipur"},
    "TRANSPORT":     {"company": "Climate-Controlled Transport","city": "Kota"},
    "DELIVERY":      {"company": "Last-Mile Delivery Hub",     "city": "Mumbai"},
}

load_dotenv()
logger = logging.getLogger(__name__)

INTERVAL = float(os.getenv("SIMULATE_INTERVAL_SECONDS", "2"))

_force_anomaly_stage: str | None = None
_injection_pending:   str | None = None
_recent_inserts: deque = deque(maxlen=40)


def request_forced_anomaly(stage: str) -> None:
    global _force_anomaly_stage, _injection_pending
    current = get_anomaly_intensity(stage)
    set_anomaly_intensity(stage, current + 1)
    _force_anomaly_stage = stage
    _injection_pending   = stage


def get_producer_state() -> dict:
    return {
        "injection_pending": _injection_pending,
        "recent_inserts":    list(_recent_inserts),
    }


async def _insert_and_log(event: dict, forced_stage: str | None = None) -> None:
    success = await db.insert_event_async(event)
    stage   = event.get("stage", "")
    meta    = _STAGE_INFO.get(stage, {})
    _recent_inserts.append({
        "ts":          datetime.now(timezone.utc).isoformat(),
        "stage":       stage,
        "company":     meta.get("company", stage),
        "city":        meta.get("city", ""),
        "temperature": round(event.get("temperature", 0), 2),
        "delay":       event.get("delay", 0),
        "inventory":   event.get("inventory", 0),
        "status":      event.get("status", "NORMAL"),
        "forced":      (stage == forced_stage),
        "ok":          success,
    })


async def run_producer() -> None:
    global _force_anomaly_stage, _injection_pending
    logger.info("Producer started - interval: %.1fs", INTERVAL)

    while True:
        try:
            forced = _force_anomaly_stage
            _force_anomaly_stage = None

            batch = generate_batch(
                anomaly_rate=float(os.getenv("ANOMALY_INJECTION_RATE", "0.02")),
                force_anomaly_stage=forced,
            )

            if forced:
                from ingestion.simulator import _generate_event
                extra_spikes = [_generate_event(forced, force_anomaly=True) for _ in range(2)]
                batch.extend(extra_spikes)

            await asyncio.gather(*[_insert_and_log(event, forced_stage=forced) for event in batch])

            if forced:
                _injection_pending = None
            else:
                decay_anomaly_intensity(decay_by=1)

        except Exception as exc:
            logger.error("Producer error: %s", exc)
            _injection_pending = None

        await asyncio.sleep(INTERVAL)