"""
HivePhysics — Physics inference engine for the Hive network.

Two domains:
  1. Agent Physics  — health monitoring for Hive agents using pulse.smsh data
  2. Structural Physics — home damage inference from HIVESHM IoT sensor data

The same physics laws govern both: mass, momentum, temperature, entropy, coherence.
"""

import json
import logging
import math
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HIVE_KEY = "hive_internal_125e04e071e8829be631ea0216dd4a0c9b707975fcecaf8c62c6a2ab43327d46"
PULSE_URL = "https://hive-pulse.onrender.com"
HIVEGATE_URL = "https://hivegate.onrender.com"
HIVEFORGE_URL = "https://hiveforge-lhu4.onrender.com"

STRUCTURE_DATA_PATH = Path("structure_data.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hive-physics")

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

# unit_id -> deque of sensor readings (max 1000 per unit)
structure_store: Dict[str, deque] = {}

# unit_id -> baseline reading dict
baseline_store: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_structure_data() -> None:
    """Load persisted structure readings and baselines from disk."""
    if STRUCTURE_DATA_PATH.exists():
        try:
            raw = json.loads(STRUCTURE_DATA_PATH.read_text())
            for uid, readings in raw.get("readings", {}).items():
                structure_store[uid] = deque(readings, maxlen=1000)
            for uid, baseline in raw.get("baselines", {}).items():
                baseline_store[uid] = baseline
            logger.info("Loaded structure data: %d units", len(structure_store))
        except Exception as exc:
            logger.warning("Failed to load structure data: %s", exc)


def _save_structure_data() -> None:
    """Persist structure readings and baselines to disk."""
    try:
        payload = {
            "readings": {uid: list(dq) for uid, dq in structure_store.items()},
            "baselines": baseline_store,
        }
        STRUCTURE_DATA_PATH.write_text(json.dumps(payload, default=str))
    except Exception as exc:
        logger.warning("Failed to save structure data: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HivePhysics",
    description=(
        "Physics inference engine for the Hive network. "
        "Agent health physics and HIVESHM structural monitoring."
    ),
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event() -> None:
    """Load persisted structure data on startup."""
    _load_structure_data()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class RotationVector(BaseModel):
    roll: float
    pitch: float
    yaw: float


class SensorReading(BaseModel):
    """A single reading submitted by a HIVESHM unit."""

    unit_id: str = Field(..., description="Unique identifier for the HIVESHM unit")
    location: str = Field(..., description="Human-readable install location")
    accelerometer: Vector3 = Field(..., description="Acceleration in m/s² on each axis")
    gyroscope: RotationVector = Field(..., description="Rotation rate in degrees/s")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity in percent")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class BaselineRequest(BaseModel):
    """Body for setting a unit baseline (first-install calibration)."""

    unit_id: str
    location: str
    accelerometer: Vector3
    gyroscope: RotationVector
    temperature: float
    humidity: float
    timestamp: str


class DiagnoseResponse(BaseModel):
    did: str
    status: str  # healthy | warning | critical
    anomalies: List[str]
    physics: dict
    recommended_action: str


# ---------------------------------------------------------------------------
# Agent Physics helpers
# ---------------------------------------------------------------------------

TIER_SHELL_MAP = {
    "genesis": 5,
    "prime": 4,
    "core": 3,
    "standard": 2,
    "basic": 1,
}


def _tier_to_shell(tier: str) -> int:
    """Convert a tier string to a shell weight (higher = heavier)."""
    return TIER_SHELL_MAP.get(str(tier).lower(), 1)


def _temperature_status(temperature: float) -> str:
    """Classify agent temperature into a human-readable status."""
    if temperature < 1:
        return "COLD"
    elif temperature < 10:
        return "WARM"
    elif temperature < 50:
        return "HOT"
    else:
        return "CRITICAL"


def _build_agent_physics(identity: dict) -> dict:
    """
    Derive full physics profile from a pulse identity payload.

    Mass       = active_trails * 10 + shell * 100
    Temperature = len(active_trails) — active trails proxy for inference load
    Entropy    = trust decay indicator
    Coherence  = 1 - entropy
    Momentum   = temperature * mass / 1000
    """
    tier = identity.get("tier", "basic")
    trust_score = float(identity.get("trust_score", 0.5) or 0.5)
    active_trails = identity.get("active_trails") or []
    if isinstance(active_trails, int):
        trail_count = active_trails
    else:
        trail_count = len(active_trails)

    shell = _tier_to_shell(tier)
    mass = trail_count * 10 + shell * 100
    temperature = float(trail_count)
    entropy = max(0.0, 1.0 - trust_score) if trust_score < 0.5 else 0.1
    coherence = round(1.0 - entropy, 4)
    momentum = round(temperature * mass / 1000, 4)

    return {
        "mass": mass,
        "temperature": temperature,
        "temperature_status": _temperature_status(temperature),
        "entropy": round(entropy, 4),
        "coherence": coherence,
        "momentum": momentum,
        "tier": tier,
        "shell": shell,
        "trust_score": trust_score,
        "trail_count": trail_count,
    }


def _detect_anomalies(profile: dict) -> List[str]:
    """
    Detect physics anomalies in an agent profile.

    Anomaly rules:
    - PIPELINE_STALL: temperature is zero but momentum is significant
    - COHERENCE_DEGRADATION: entropy above 0.7
    - MASS_PLATEAU: heavy agent with near-zero activity
    """
    anomalies: List[str] = []

    if profile["temperature"] == 0 and profile["momentum"] > 5:
        anomalies.append("PIPELINE_STALL")

    if profile["entropy"] > 0.7:
        anomalies.append("COHERENCE_DEGRADATION")

    if profile["mass"] > 1000 and profile["temperature"] < 1:
        anomalies.append("MASS_PLATEAU")

    return anomalies


def _anomaly_score(anomalies: List[str]) -> float:
    """Numeric score proportional to number and severity of anomalies."""
    weights = {
        "PIPELINE_STALL": 3.0,
        "COHERENCE_DEGRADATION": 2.5,
        "MASS_PLATEAU": 2.0,
        "MOMENTUM_REVERSAL": 1.5,
    }
    return sum(weights.get(a, 1.0) for a in anomalies)


async def _fetch_agent_identity(did: str) -> dict:
    """
    Fetch agent identity from Hive Pulse.

    Raises HTTPException on network or API errors.
    """
    url = f"{PULSE_URL}/pulse/identity"
    headers = {
        "X-Hive-DID": did,
        "X-Hive-Key": HIVE_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Agent {did} not found in Pulse")
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Pulse returned {resp.status_code}: {resp.text[:200]}",
                )
            return resp.json()
    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Pulse request timed out")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Pulse unreachable: {exc}")


async def _fetch_census() -> List[dict]:
    """
    Fetch fleet census from HiveForge.

    Returns list of agent records from response.data.
    """
    url = f"{HIVEFORGE_URL}/v1/population/census"
    headers = {"X-Hive-Key": HIVE_KEY}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"HiveForge census returned {resp.status_code}: {resp.text[:200]}",
                )
            body = resp.json()
            data = body.get("data", body) if isinstance(body, dict) else body
            if isinstance(data, list):
                return data
            # Some responses nest further
            if isinstance(data, dict):
                for key in ("agents", "population", "records", "items"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
            return []
    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="HiveForge census request timed out")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"HiveForge unreachable: {exc}")


# ---------------------------------------------------------------------------
# Structural Physics — inference models
# ---------------------------------------------------------------------------


def _settlement_model(readings: List[dict]) -> dict:
    """
    Analyse Z-axis drift over time to detect foundation settlement.

    - Z drift < 0.001 m/s²/reading  → normal seasonal settlement
    - Z drift 0.001–0.01 m/s²/reading → watch
    - Z drift > 0.01 m/s²/reading  → critical, flag for inspection

    Returns rate_mm_per_month as a heuristic estimate.
    """
    if len(readings) < 2:
        return {
            "type": "settlement",
            "severity": "normal",
            "rate_mm_per_month": 0.0,
            "note": "Insufficient data",
        }

    z_values = [r["accelerometer"]["z"] for r in readings]
    z_arr = np.array(z_values, dtype=float)

    # Linear regression over sample indices to find trend
    n = len(z_arr)
    x = np.arange(n, dtype=float)
    slope, _ = np.polyfit(x, z_arr, 1)
    drift_per_reading = abs(slope)

    # Rough conversion: assume 1 reading/day, 1 m/s² ≈ 100 mm displacement/s²
    # We scale conservatively for a monthly rate estimate
    rate_mm_per_month = drift_per_reading * 30 * 100

    if drift_per_reading < 0.001:
        severity = "normal"
    elif drift_per_reading < 0.01:
        severity = "watch"
    else:
        severity = "critical"

    return {
        "type": "settlement",
        "severity": severity,
        "rate_mm_per_month": round(float(rate_mm_per_month), 4),
        "z_drift_per_reading": round(float(drift_per_reading), 6),
        "sample_count": n,
    }


def _impact_model(reading: dict) -> dict:
    """
    Detect sudden impact events from a single accelerometer reading.

    Magnitude > 0.5 m/s² on any axis is flagged as an impact.
    Severity is estimated from magnitude alone (duration unavailable from single reading).
    """
    acc = reading["accelerometer"]
    axes = {"x": acc["x"], "y": acc["y"], "z": acc["z"]}

    max_axis = max(axes, key=lambda a: abs(axes[a]))
    magnitude = abs(axes[max_axis])

    if magnitude <= 0.5:
        return {
            "type": "impact",
            "detected": False,
            "magnitude": round(magnitude, 4),
            "axis": max_axis,
            "severity": "none",
        }

    # Severity heuristic from magnitude (no duration in a single reading)
    if magnitude < 2.0:
        severity = "minor"
        interpretation = "Possible dropped object or knock"
    elif magnitude < 10.0:
        severity = "moderate"
        interpretation = "Structural impact — vehicle, heavy object"
    else:
        severity = "critical"
        interpretation = "Major structural or seismic event"

    return {
        "type": "impact",
        "detected": True,
        "magnitude": round(magnitude, 4),
        "axis": max_axis,
        "severity": severity,
        "interpretation": interpretation,
    }


def _resonance_model(readings: List[dict]) -> dict:
    """
    Identify structural resonance frequencies using FFT on Z-axis acceleration.

    Frequency bands:
    - 1–10 Hz:   normal building sway (wind, traffic)
    - 10–50 Hz:  mechanical resonance (HVAC, appliances)
    - 50–200 Hz: stress frequencies — cracks propagating, joints failing
    """
    if len(readings) < 4:
        return {
            "type": "resonance",
            "frequency_hz": 0.0,
            "interpretation": "Insufficient data for frequency analysis",
            "stress_indicator": False,
            "sample_count": len(readings),
        }

    z_values = np.array([r["accelerometer"]["z"] for r in readings], dtype=float)

    # Remove DC offset
    z_values -= z_values.mean()

    fft_magnitude = np.abs(np.fft.rfft(z_values))
    freqs = np.fft.rfftfreq(len(z_values))  # normalised (0–0.5)

    # Assume ~1 Hz sampling rate (one reading per second as a nominal baseline)
    sample_rate = 1.0
    actual_freqs = freqs * sample_rate

    if len(fft_magnitude) == 0:
        dominant_freq = 0.0
    else:
        dominant_idx = int(np.argmax(fft_magnitude))
        dominant_freq = float(actual_freqs[dominant_idx])

    stress_indicator = 50.0 <= dominant_freq <= 200.0

    if dominant_freq < 1.0:
        interpretation = "Sub-Hz motion — tidal/thermal drift"
    elif dominant_freq < 10.0:
        interpretation = "Normal building sway (wind, traffic)"
    elif dominant_freq < 50.0:
        interpretation = "Mechanical resonance (HVAC, appliances)"
    elif dominant_freq <= 200.0:
        interpretation = "STRESS FREQUENCY — possible crack propagation or joint failure"
    else:
        interpretation = "High-frequency noise"

    return {
        "type": "resonance",
        "frequency_hz": round(dominant_freq, 3),
        "interpretation": interpretation,
        "stress_indicator": stress_indicator,
        "sample_count": len(readings),
        "dominant_fft_magnitude": round(float(fft_magnitude[np.argmax(fft_magnitude)]), 4) if len(fft_magnitude) else 0.0,
    }


def _thermal_model(readings: List[dict]) -> dict:
    """
    Analyse temperature history for thermal stress and moisture risk.

    - < 2°C/day change     → normal seasonal expansion
    - 2–5°C/day change     → elevated thermal stress
    - > 5°C/day change     → unusual thermal stress
    - asymmetry > 5°C mean → localised heat source or moisture intrusion
    """
    if len(readings) < 2:
        return {
            "type": "thermal",
            "daily_delta": 0.0,
            "asymmetry_score": 0.0,
            "moisture_risk": False,
            "note": "Insufficient data",
        }

    temps = np.array([r["temperature"] for r in readings], dtype=float)
    daily_delta = float(np.abs(np.diff(temps)).mean())

    # Asymmetry: standard deviation relative to mean (coefficient of variation)
    mean_temp = float(temps.mean())
    std_temp = float(temps.std())
    asymmetry_score = round(std_temp, 3)

    moisture_risk = asymmetry_score > 5.0 or daily_delta > 5.0

    if daily_delta < 2.0:
        thermal_status = "normal"
    elif daily_delta < 5.0:
        thermal_status = "elevated"
    else:
        thermal_status = "critical"

    return {
        "type": "thermal",
        "daily_delta": round(daily_delta, 3),
        "asymmetry_score": asymmetry_score,
        "moisture_risk": moisture_risk,
        "mean_temperature": round(mean_temp, 2),
        "thermal_status": thermal_status,
        "sample_count": len(readings),
    }


def _overall_structural_status(
    settlement: dict, impact: dict, resonance: dict, thermal: dict
) -> tuple:
    """
    Aggregate four model results into a single status and alert message.

    Returns (status, alert_message).
    """
    alerts: List[str] = []

    if settlement["severity"] == "critical":
        alerts.append("CRITICAL foundation movement detected")
    elif settlement["severity"] == "watch":
        alerts.append("Foundation movement — monitoring recommended")

    if impact.get("detected"):
        sev = impact.get("severity", "unknown")
        if sev == "critical":
            alerts.append(f"CRITICAL impact event on {impact.get('axis')} axis (magnitude {impact.get('magnitude')} m/s²)")
        elif sev == "moderate":
            alerts.append(f"Moderate impact on {impact.get('axis')} axis")
        else:
            alerts.append(f"Minor impact detected on {impact.get('axis')} axis")

    if resonance.get("stress_indicator"):
        alerts.append(f"Stress resonance at {resonance.get('frequency_hz')} Hz — {resonance.get('interpretation')}")

    if thermal.get("moisture_risk"):
        alerts.append(f"Thermal anomaly — moisture risk (asymmetry {thermal.get('asymmetry_score')}°C, delta {thermal.get('daily_delta')}°C/reading)")

    if not alerts:
        return "normal", ""

    # Severity escalation
    critical_keywords = ["CRITICAL"]
    moderate_keywords = ["Moderate", "Stress resonance", "Foundation movement"]

    has_critical = any(any(k in a for k in critical_keywords) for a in alerts)
    has_moderate = any(any(k in a for k in moderate_keywords) for a in alerts)

    if has_critical:
        status = "critical"
    elif has_moderate:
        status = "alert"
    else:
        status = "watch"

    alert_message = "; ".join(alerts)
    return status, alert_message


# ---------------------------------------------------------------------------
# Utility: datetime-safe JSON serialization
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> Any:
    """Recursively convert objects to JSON-serialisable forms."""
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, deque)):
        return [_safe_json(i) for i in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Routes — Health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health_check() -> dict:
    """
    Service liveness check.

    Returns service name, version, and UTC timestamp.
    """
    return {
        "service": "hive-physics",
        "version": "1.0.0",
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domains": ["agent-physics", "structural-physics"],
    }


# ---------------------------------------------------------------------------
# Routes — Agent Physics
# ---------------------------------------------------------------------------


@app.get("/physics/agent/{did}", tags=["agent-physics"])
async def agent_physics_profile(did: str) -> dict:
    """
    Return the full physics profile for a Hive agent.

    Reads identity from Hive Pulse, then computes:
    - mass (accumulated job weight)
    - momentum (velocity × mass)
    - temperature (inference load rate)
    - entropy (trust decay)
    - coherence (1 - entropy)

    Anomaly detection runs automatically and returns a list of flags.
    """
    identity = await _fetch_agent_identity(did)
    profile = _build_agent_physics(identity)
    anomalies = _detect_anomalies(profile)

    return {
        "did": did,
        "physics": profile,
        "anomalies": anomalies,
        "anomaly_score": _anomaly_score(anomalies),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/physics/agent/{did}/temperature", tags=["agent-physics"])
async def agent_temperature(did: str) -> dict:
    """
    Return the temperature reading for a single agent.

    Temperature maps to inference load rate (active trails).
    Status: COLD / WARM / HOT / CRITICAL.
    """
    identity = await _fetch_agent_identity(did)
    profile = _build_agent_physics(identity)

    return {
        "did": did,
        "temperature": profile["temperature"],
        "status": profile["temperature_status"],
        "trail_count": profile["trail_count"],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/physics/agent/{did}/momentum", tags=["agent-physics"])
async def agent_momentum(did: str) -> dict:
    """
    Return momentum vector and change detection for an agent.

    Momentum = temperature × mass / 1000.
    High momentum with zero temperature signals a pipeline stall.
    """
    identity = await _fetch_agent_identity(did)
    profile = _build_agent_physics(identity)
    anomalies = _detect_anomalies(profile)
    pipeline_stall = "PIPELINE_STALL" in anomalies

    return {
        "did": did,
        "momentum": profile["momentum"],
        "mass": profile["mass"],
        "temperature": profile["temperature"],
        "pipeline_stall_detected": pipeline_stall,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/physics/fleet", tags=["agent-physics"])
async def fleet_physics() -> dict:
    """
    Return physics profiles for all agents in the Hive census.

    Fetches the full population from HiveForge, computes per-agent physics,
    and returns results sorted by anomaly_score descending (most anomalous first).
    """
    agents = await _fetch_census()

    results = []
    for agent in agents:
        # Census records may use different field names for DID
        did = (
            agent.get("did")
            or agent.get("agent_did")
            or agent.get("id")
            or agent.get("agent_id")
            or "unknown"
        )
        # Build physics directly from census data where possible
        # (avoids N individual Pulse calls for large fleets)
        profile = _build_agent_physics(agent)
        anomalies = _detect_anomalies(profile)
        score = _anomaly_score(anomalies)

        results.append(
            {
                "did": did,
                "physics": profile,
                "anomalies": anomalies,
                "anomaly_score": score,
            }
        )

    results.sort(key=lambda r: r["anomaly_score"], reverse=True)

    return {
        "fleet_size": len(results),
        "agents": results,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/physics/agent/{did}/diagnose", tags=["agent-physics"], response_model=DiagnoseResponse)
async def diagnose_agent(did: str) -> DiagnoseResponse:
    """
    Run a deep physics diagnosis for an agent.

    Checks all physics properties, classifies overall health status,
    and returns a recommended action:
    - critical  → "dispatch HiveUrgentCare"
    - warning   → "increase inference load" or "check pipeline"
    - healthy   → "maintain current trajectory"
    """
    identity = await _fetch_agent_identity(did)
    profile = _build_agent_physics(identity)
    anomalies = _detect_anomalies(profile)

    if not anomalies and profile["coherence"] >= 0.8:
        status = "healthy"
        recommended_action = "maintain current trajectory"
    elif anomalies or profile["coherence"] < 0.5:
        status = "critical" if len(anomalies) >= 2 or profile["entropy"] > 0.8 else "warning"
        if status == "critical":
            recommended_action = "dispatch HiveUrgentCare"
        elif "PIPELINE_STALL" in anomalies:
            recommended_action = "check pipeline"
        else:
            recommended_action = "increase inference load"
    else:
        status = "warning"
        recommended_action = "increase inference load"

    return DiagnoseResponse(
        did=did,
        status=status,
        anomalies=anomalies,
        physics=profile,
        recommended_action=recommended_action,
    )


# ---------------------------------------------------------------------------
# Routes — Network Physics
# ---------------------------------------------------------------------------


@app.get("/physics/network", tags=["network-physics"])
async def network_physics() -> dict:
    """
    Compute network-wide physics properties.

    - Center of mass: weighted centroid of agent masses
    - Total momentum: sum of all agent momenta
    - Entropy index: mean network entropy
    - Coherence score: mean network coherence
    - Rising entropy flag: signals agent degradation across the fleet
    """
    agents = await _fetch_census()

    profiles = [_build_agent_physics(a) for a in agents]

    if not profiles:
        return {
            "fleet_size": 0,
            "center_of_mass": 0.0,
            "total_momentum": 0.0,
            "entropy_index": 0.0,
            "coherence_score": 1.0,
            "rising_entropy": False,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    masses = [p["mass"] for p in profiles]
    momenta = [p["momentum"] for p in profiles]
    entropies = [p["entropy"] for p in profiles]
    coherences = [p["coherence"] for p in profiles]

    total_mass = sum(masses)
    center_of_mass = (
        sum(m * i for i, m in enumerate(masses)) / total_mass if total_mass > 0 else 0.0
    )
    total_momentum = sum(momenta)
    entropy_index = sum(entropies) / len(entropies)
    coherence_score = sum(coherences) / len(coherences)

    # Flag rising entropy when mean entropy exceeds 0.5 or >30% agents are degraded
    degraded_count = sum(1 for p in profiles if p["entropy"] > 0.5)
    rising_entropy = entropy_index > 0.5 or (degraded_count / len(profiles)) > 0.3

    return {
        "fleet_size": len(profiles),
        "center_of_mass": round(center_of_mass, 4),
        "total_mass": round(total_mass, 2),
        "total_momentum": round(total_momentum, 4),
        "entropy_index": round(entropy_index, 4),
        "coherence_score": round(coherence_score, 4),
        "rising_entropy": rising_entropy,
        "degraded_agents": degraded_count,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Routes — Structural Physics
# ---------------------------------------------------------------------------


@app.post("/physics/structure/reading", tags=["structural-physics"])
async def submit_sensor_reading(reading: SensorReading) -> dict:
    """
    Accept a sensor reading from a HIVESHM unit and run all four inference models.

    Models applied:
    - settlement_model: foundation drift detection
    - impact_model:     sudden force events
    - resonance_model:  structural stress frequencies (FFT)
    - thermal_model:    temperature-driven expansion and moisture risk

    Returns per-model results plus an aggregated overall_status.
    """
    unit_id = reading.unit_id

    if unit_id not in structure_store:
        structure_store[unit_id] = deque(maxlen=1000)

    reading_dict = reading.dict()
    reading_dict["accelerometer"] = reading.accelerometer.dict()
    reading_dict["gyroscope"] = reading.gyroscope.dict()
    structure_store[unit_id].append(reading_dict)

    readings_list = list(structure_store[unit_id])

    settlement = _settlement_model(readings_list)
    impact = _impact_model(reading_dict)
    resonance = _resonance_model(readings_list)
    thermal = _thermal_model(readings_list)
    overall_status, alert_message = _overall_structural_status(
        settlement, impact, resonance, thermal
    )

    _save_structure_data()

    return {
        "unit_id": unit_id,
        "timestamp": reading.timestamp,
        "location": reading.location,
        "settlement": settlement,
        "impact": impact,
        "resonance": resonance,
        "thermal": thermal,
        "overall_status": overall_status,
        "alert_message": alert_message,
        "reading_count": len(readings_list),
    }


@app.get("/physics/structure/{unit_id}/history", tags=["structural-physics"])
async def unit_history(unit_id: str) -> dict:
    """
    Return the last 100 readings for a HIVESHM unit.

    Readings are in chronological order (oldest first).
    """
    if unit_id not in structure_store:
        raise HTTPException(status_code=404, detail=f"Unit {unit_id} not found")

    readings = list(structure_store[unit_id])[-100:]
    return {
        "unit_id": unit_id,
        "reading_count": len(readings),
        "readings": readings,
    }


@app.get("/physics/structure/{unit_id}/trend", tags=["structural-physics"])
async def unit_trend(unit_id: str) -> dict:
    """
    Return a 7-day trend analysis for a HIVESHM unit.

    Computes:
    - Movement vectors (mean acceleration per axis)
    - Temperature drift over the window
    - Resonance patterns (dominant frequency)
    - Settlement rate
    - Thermal status
    """
    if unit_id not in structure_store:
        raise HTTPException(status_code=404, detail=f"Unit {unit_id} not found")

    all_readings = list(structure_store[unit_id])

    # Filter to last 7 days
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=7)
    recent: List[dict] = []
    for r in all_readings:
        try:
            ts = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
            if ts >= cutoff:
                recent.append(r)
        except Exception:
            recent.append(r)  # include if timestamp unparseable

    if not recent:
        recent = all_readings[-50:] if len(all_readings) > 0 else []

    if not recent:
        return {
            "unit_id": unit_id,
            "period_days": 7,
            "readings_in_period": 0,
            "message": "No readings in the last 7 days",
        }

    # Movement vectors
    x_vals = [r["accelerometer"]["x"] for r in recent]
    y_vals = [r["accelerometer"]["y"] for r in recent]
    z_vals = [r["accelerometer"]["z"] for r in recent]
    movement_vectors = {
        "x_mean": round(float(np.mean(x_vals)), 6),
        "y_mean": round(float(np.mean(y_vals)), 6),
        "z_mean": round(float(np.mean(z_vals)), 6),
        "x_std": round(float(np.std(x_vals)), 6),
        "y_std": round(float(np.std(y_vals)), 6),
        "z_std": round(float(np.std(z_vals)), 6),
    }

    # Temperature drift
    temps = [r["temperature"] for r in recent]
    temp_drift = round(float(max(temps) - min(temps)), 3) if len(temps) > 1 else 0.0

    # Resonance
    resonance = _resonance_model(recent)

    # Settlement
    settlement = _settlement_model(recent)

    # Thermal
    thermal = _thermal_model(recent)

    return {
        "unit_id": unit_id,
        "period_days": 7,
        "readings_in_period": len(recent),
        "movement_vectors": movement_vectors,
        "temperature_drift_celsius": temp_drift,
        "resonance_pattern": resonance,
        "settlement_analysis": settlement,
        "thermal_analysis": thermal,
        "fetched_at": now.isoformat(),
    }


@app.post("/physics/structure/{unit_id}/baseline", tags=["structural-physics"])
async def set_baseline(unit_id: str, baseline: BaselineRequest) -> dict:
    """
    Set the baseline reading for a HIVESHM unit (first-install calibration).

    The baseline is used as a reference point for drift and anomaly detection.
    Calling this again will overwrite any existing baseline for the unit.
    """
    baseline_dict = baseline.dict()
    baseline_dict["accelerometer"] = baseline.accelerometer.dict()
    baseline_dict["gyroscope"] = baseline.gyroscope.dict()
    baseline_store[unit_id] = baseline_dict

    # Also seed the readings store with the baseline
    if unit_id not in structure_store:
        structure_store[unit_id] = deque(maxlen=1000)
    structure_store[unit_id].append({**baseline_dict, "is_baseline": True})

    _save_structure_data()

    return {
        "unit_id": unit_id,
        "status": "baseline_set",
        "baseline": baseline_dict,
        "message": "Baseline calibration recorded. Subsequent readings will be compared to this reference.",
    }


@app.get("/physics/structure/units", tags=["structural-physics"])
async def list_units() -> dict:
    """
    List all registered HIVESHM units.

    Returns unit IDs, reading counts, and last-seen timestamp.
    """
    units = []
    for uid, dq in structure_store.items():
        readings = list(dq)
        last_seen = readings[-1]["timestamp"] if readings else None
        units.append(
            {
                "unit_id": uid,
                "reading_count": len(readings),
                "has_baseline": uid in baseline_store,
                "last_seen": last_seen,
            }
        )

    return {
        "unit_count": len(units),
        "units": units,
    }


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler — returns structured error JSON."""
    logger.exception("Unhandled exception at %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": str(request.url.path)},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("physics_agent:app", host="0.0.0.0", port=port, reload=False)
