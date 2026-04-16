from __future__ import annotations
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import db.griddb_client as db_client
from db.schema import STAGES
from ingestion.producer import run_producer, request_forced_anomaly, get_producer_state
from ingestion.simulator import get_anomaly_intensity
from llm.reasoning import get_insights
from ml.trainer import train_if_needed
from risk.cascade import propagate_risk
from risk.risk_engine import compute_all_stage_risks
from db.supply_chain_config import SCENARIO, STAGE_META

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_latest_payload: dict = {}
_insights_cache:  dict = {}
_insights_lock  = asyncio.Lock()
_INSIGHT_REFRESH_EVERY = 3


async def _pipeline_worker() -> None:
    """Computes risk and LLM insights in a background loop."""
    global _latest_payload, _insights_cache
    cycle = 0
    interval = float(os.getenv("SIMULATE_INTERVAL_SECONDS", "2"))

    while True:
        try:
            stage_events = db_client.get_all_stage_events(minutes=10)
            stage_risks  = compute_all_stage_risks(stage_events)
            cascade      = propagate_risk(stage_risks)

            all_recent = []
            for evts in stage_events.values():
                all_recent.extend(evts[-6:])
            all_recent.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

            _latest_payload = {
                "timestamp":    datetime.now(timezone.utc).isoformat(),
                "stage_risks":  stage_risks,
                "cascade":      cascade,
                "recent_events": all_recent[:30],
            }

            if cycle % _INSIGHT_REFRESH_EVERY == 0:
                async with _insights_lock:
                    loop = asyncio.get_event_loop()
                    _insights_cache = await loop.run_in_executor(
                        None, get_insights, stage_risks, cascade
                    )
            cycle += 1
        except Exception as exc:
            logger.error("Pipeline worker error: %s", exc)
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_client.initialize()
    train_if_needed()
    asyncio.create_task(run_producer())
    asyncio.create_task(_pipeline_worker())
    logger.info("Supply Chain AI system started.")
    yield


app = FastAPI(
    title="Supply Chain AI",
    description="AI-Powered Multi-Stage Supply Chain Intelligence System",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/api/status")
async def status():
    if not _latest_payload:
        return {"status": "warming_up", "message": "Pipeline initialising…"}
    c = _latest_payload.get("cascade", {})
    return {
        "status":      "ok",
        "global_risk": c.get("global_risk", 0),
        "risk_level":  c.get("risk_level", "UNKNOWN"),
        "timestamp":   _latest_payload.get("timestamp"),
    }


@app.get("/api/stages")
async def stages():
    if not _latest_payload:
        return {"stages": {}}
    return {
        "stages":   _latest_payload.get("stage_risks", {}),
        "cascade":  _latest_payload.get("cascade", {}),
        "timestamp": _latest_payload.get("timestamp"),
    }


@app.get("/api/events")
async def events(
    stage:   str | None = Query(None),
    minutes: int        = Query(10, ge=1, le=60),
):
    return {"events": db_client.query_recent(stage=stage, minutes=minutes)}


@app.get("/api/insights")
async def insights():
    async with _insights_lock:
        if not _insights_cache:
            return {"message": "Insights not yet generated."}
        return _insights_cache


@app.post("/api/simulate/anomaly")
async def force_anomaly(stage: str = Query(..., description="Stage to inject anomaly into")):
    stage = stage.upper()
    if stage not in STAGES:
        return {"error": f"Unknown stage: {stage}"}
    request_forced_anomaly(stage)
    return {"injected": True, "stage": stage}


@app.get("/api/snapshot")
async def snapshot():
    if not _latest_payload:
        return JSONResponse({"ready": False})
    async with _insights_lock:
        stage_insights = _insights_cache.get("stage_insights", {})
    producer = get_producer_state()
    intensity_map = {stage: get_anomaly_intensity(stage) for stage in STAGES}
    return JSONResponse({
        "ready":             True,
        "stage_risks":       _latest_payload.get("stage_risks", {}),
        "cascade":           _latest_payload.get("cascade", {}),
        "recent_events":     _latest_payload.get("recent_events", []),
        "timestamp":         _latest_payload.get("timestamp"),
        "scenario":          SCENARIO,
        "stage_meta":        STAGE_META,
        "stage_insights":    stage_insights,
        "recent_inserts":    producer["recent_inserts"],
        "injection_pending": producer["injection_pending"],
        "anomaly_intensity": intensity_map,
    })


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)