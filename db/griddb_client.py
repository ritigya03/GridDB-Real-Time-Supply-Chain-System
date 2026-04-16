from __future__ import annotations
import logging
import os
import httpx
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any
from dotenv import load_dotenv
from db.schema import CONTAINER_NAME, CONTAINER_SCHEMA, STAGES

load_dotenv()
logger = logging.getLogger(__name__)

GRIDDB_HOST     = os.getenv("GRIDDB_HOST", "")
GRIDDB_PORT     = int(os.getenv("GRIDDB_PORT", "443"))
GRIDDB_CLUSTER  = os.getenv("GRIDDB_CLUSTER", "")
GRIDDB_DATABASE = os.getenv("GRIDDB_DATABASE", "public")
GRIDDB_USER     = os.getenv("GRIDDB_USER", "")
GRIDDB_PASSWORD = os.getenv("GRIDDB_PASSWORD", "")

_BASE_URL = f"https://{GRIDDB_HOST}:{GRIDDB_PORT}/griddb/v2/{GRIDDB_CLUSTER}/dbs/{GRIDDB_DATABASE}"
_async_client: httpx.AsyncClient | None = None


def _ts_to_str(ts: datetime) -> str:
    """Serialize datetime to GridDB-compatible ISO-8601 string."""
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _auth() -> httpx.BasicAuth:
    return httpx.BasicAuth(GRIDDB_USER, GRIDDB_PASSWORD)


def _get_async_client() -> httpx.AsyncClient:
    """Return a lazily-initialised, connection-pooled AsyncClient."""
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            auth=_auth(),
            timeout=10,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _async_client


def _ensure_container() -> None:
    """Create the container if it doesn't exist."""
    url = f"{_BASE_URL}/containers"
    try:
        resp = httpx.post(url, json=CONTAINER_SCHEMA, auth=_auth(), timeout=15)
        if resp.status_code in (200, 201):
            logger.info("GridDB container '%s' created.", CONTAINER_NAME)
        elif resp.status_code != 409:
            logger.warning("Container status %s: %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.error("Failed to ensure GridDB container: %s", exc)


def insert_event(event: dict[str, Any]) -> bool:
    """Persist a single supply-chain event to GridDB Cloud."""
    ts = _ts_to_str(datetime.now(timezone.utc))
    row = [
        ts, event["entity_id"], event["stage"],
        float(event["temperature"]), int(event["delay"]),
        int(event["inventory"]), event["status"]
    ]

    url = f"{_BASE_URL}/containers/{CONTAINER_NAME}/rows"
    try:
        resp = httpx.put(url, json=[row], auth=_auth(), timeout=10)
        return resp.status_code in (200, 201)
    except Exception as exc:
        logger.error("Insert failed: %s", exc)
        return False


async def insert_event_async(event: dict[str, Any]) -> bool:
    """Non-blocking event insertion using shared AsyncClient."""
    ts  = _ts_to_str(datetime.now(timezone.utc))
    row = [
        ts, event["entity_id"], event["stage"],
        float(event["temperature"]), int(event["delay"]),
        int(event["inventory"]), event["status"]
    ]
    url = f"{_BASE_URL}/containers/{CONTAINER_NAME}/rows"
    try:
        client = _get_async_client()
        resp   = await client.put(url, json=[row])
        return resp.status_code in (200, 201)
    except Exception:
        return False


def query_recent(stage: str | None = None, minutes: int = 10) -> list[dict]:
    """Fetch events from recent time window using TQL."""
    ts_str = _ts_to_str(datetime.now(timezone.utc) - timedelta(minutes=minutes))
    tql = f"SELECT * WHERE timestamp > TIMESTAMP('{ts_str}')"
    if stage:
        tql += f" AND stage = '{stage}'"

    url  = f"{_BASE_URL}/tql"
    body = [{"name": CONTAINER_NAME, "stmt": tql + " LIMIT 5000"}]
    try:
        resp = httpx.post(url, json=body, auth=_auth(), timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            columns = [c["name"] for c in CONTAINER_SCHEMA["columns"]]
            rows = data[0].get("results", []) if data else []
            return [dict(zip(columns, row)) for row in rows]
    except Exception as exc:
        logger.error("Query failed: %s", exc)
    return []


def get_all_stage_events(minutes: int = 10) -> dict[str, list[dict]]:
    """Return recent events grouped by stage."""
    all_events = query_recent(stage=None, minutes=minutes)
    grouped: dict[str, list[dict]] = {s: [] for s in STAGES}
    for e in all_events:
        s = e.get("stage", "")
        if s in grouped:
            grouped[s].append(e)
    return grouped


def initialize() -> None:
    missing = [var for var, val in {
        "GRIDDB_HOST": GRIDDB_HOST, "GRIDDB_CLUSTER": GRIDDB_CLUSTER,
        "GRIDDB_DATABASE": GRIDDB_DATABASE, "GRIDDB_USER": GRIDDB_USER,
        "GRIDDB_PASSWORD": GRIDDB_PASSWORD
    }.items() if not val]

    if missing:
        raise RuntimeError(f"Missing config: {', '.join(missing)}")

    logger.info("GridDB client initialized.")
    _ensure_container()
