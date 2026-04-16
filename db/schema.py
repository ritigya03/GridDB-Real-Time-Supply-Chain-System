CONTAINER_NAME = "supply_chain_events"

CONTAINER_SCHEMA = {
    "container_name": CONTAINER_NAME,
    "container_type": "TIME_SERIES",
    "rowkey": True,
    "columns": [
        {"name": "timestamp",   "type": "TIMESTAMP"},
        {"name": "entity_id",   "type": "STRING"},
        {"name": "stage",       "type": "STRING"},
        {"name": "temperature", "type": "DOUBLE"},
        {"name": "delay",       "type": "INTEGER"},
        {"name": "inventory",   "type": "INTEGER"},
        {"name": "status",      "type": "STRING"},
    ],
}

STAGES = ["SUPPLIER", "MANUFACTURING", "WAREHOUSE", "TRANSPORT", "DELIVERY"]

STATUS_NORMAL  = "NORMAL"
STATUS_WARNING = "WARNING"
STATUS_ANOMALY = "ANOMALY"
