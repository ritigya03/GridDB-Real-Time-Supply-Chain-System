# GridDB Real-Time Supply Chain AI

A real-time monitoring and intelligence system for multi-stage supply chains. Powered by **GridDB Cloud** for time-series persistence, **Scikit-Learn** for anomaly detection, and **LLM** for automated reasoning.

## Project Structure

```text
├── api/
│   ├── static/         # Dashboard UI (index.html)
│   └── app.py         # FastAPI backend & Pipeline worker
├── db/
│   ├── griddb_client.py   # REST API client for GridDB Cloud
│   ├── schema.py          # Container & Stage definitions
│   └── supply_chain_config.py # Scenario context & Metadata
├── features/
│   └── feature_engine.py  # Time-series feature extraction
├── ingestion/
│   ├── producer.py        # Asynchronous data ingestion loop
│   └── simulator.py       # Multi-stage event simulator
├── llm/
│   └── reasoning.py       # Groq AI insight generator
├── ml/
│   ├── anomaly_model.py   # Isolation Forest wrapper
│   └── trainer.py         # Model training & persistence
├── risk/
│   ├── cascade.py         # Risk propagation engine
│   └── risk_engine.py      # Local stage-wise risk computation
├── main.py                # Application entry point
├── requirements.txt       # Project dependencies
└── .env.example          # Template for environment variables
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Create a `.env` file with your **GridDB Cloud** and **Groq API** credentials.

3. **Launch System**:
   ```bash
   uvicorn main:app --reload
   ```
   Open `http://localhost:8000/` in your browser.

## ⚙️ Tech Stack
- **Persistence**: GridDB Cloud (Time-Series)
- **Intelligence**: Llama 3.1 (via Groq), Isolation Forest (ML)
- **Backend**: FastAPI (Python)
- **UI**: Vanilla JS / CSS3