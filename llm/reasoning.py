from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_groq_client = None


def _get_client():
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
        return _groq_client
    except Exception as exc:
        logger.warning("Groq client init failed: %s", exc)
        return None


def _build_prompt(stage_risks: dict[str, dict], cascade: dict) -> str:
    lines = [
        "You are an expert supply chain intelligence system.",
        "Analyse the following real-time risk data and respond in JSON.\n",
        f"Global Risk Score: {cascade['global_risk']}/100 ({cascade['risk_level']})",
        f"Highest Risk Stage: {cascade['highest_risk_stage']}\n",
        "Stage-by-Stage Risk Breakdown:"
    ]
    for stage, risk_dict in stage_risks.items():
        f = risk_dict.get("features", {})
        lines.append(
            f"  {stage}: risk={risk_dict['risk_score']:.1f}/100 | "
            f"anomaly={'YES' if risk_dict['is_anomaly'] else 'no'} | "
            f"mean_temp={f.get('mean_temp', 0):.1f}°C | "
            f"mean_delay={f.get('mean_delay', 0):.1f}h"
        )

    lines.append(
        "\nRespond ONLY with valid JSON:\n"
        "{\n"
        '  "risk_level": "<LOW|MEDIUM|HIGH|CRITICAL>",\n'
        '  "summary": "<1-2 sentence overall assessment>",\n'
        '  "predicted_outcomes": ["<outcome 1>", "<outcome 2>", "<outcome 3>"],\n'
        '  "suggested_actions": ["<action 1>", "<action 2>", "<action 3>"],\n'
        '  "stage_insights": {\n'
        '    "<STAGE>": "<1 sentence stage-specific insight>"\n'
        '  }\n'
        "}"
    )
    return "\n".join(lines)


def _rule_based_response(cascade: dict, stage_risks: dict[str, dict]) -> dict:
    level = cascade["risk_level"]
    highest = cascade["highest_risk_stage"]

    outcomes = {
        "LOW":      ["Operating normally", "No immediate intervention required"],
        "MEDIUM":   ["Potential delay build-up", "Monitor transport closely"],
        "HIGH":     ["Risk of stockouts", "Temperature exceedance risk", "Cascade delays likely"],
        "CRITICAL": ["Immediate disruption imminent", "Critical inventory shortage"],
    }
    actions = {
        "LOW":      ["Continue standard monitoring", "Maintain buffer stock"],
        "MEDIUM":   ["Activate backup supplier", "Alert logistics team"],
        "HIGH":     ["Reroute shipments", "Emergency restocking order"],
        "CRITICAL": ["Invoke contingency plan", "Emergency procurement"],
    }

    stage_insights = {
        s: f"{s} risk at {r['risk_score']:.0f}/100 — {'anomaly detected' if r['is_anomaly'] else 'normal'}."
        for s, r in stage_risks.items()
    }

    return {
        "risk_level":        level,
        "summary":           f"Supply chain at {level} risk. Impact primarily at {highest}.",
        "predicted_outcomes": outcomes.get(level, outcomes["MEDIUM"]),
        "suggested_actions":  actions.get(level, actions["MEDIUM"]),
        "stage_insights":    stage_insights,
        "source":            "rule-based",
    }


def get_insights(stage_risks: dict[str, dict], cascade: dict) -> dict:
    """Generate LLM-powered insights with rule-based fallback."""
    client = _get_client()
    if client is None:
        result = _rule_based_response(cascade, stage_risks)
        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        return result

    prompt = _build_prompt(stage_risks, cascade)
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        raw_text = response.choices[0].message.content.strip()
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"): raw_text = raw_text[4:]

        parsed = json.loads(raw_text)
        parsed["source"] = "groq-llm"
        parsed["generated_at"] = datetime.now(timezone.utc).isoformat()
        return parsed
    except Exception:
        result = _rule_based_response(cascade, stage_risks)
        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        return result
