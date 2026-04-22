# HivePhysics

Physics inference engine for the Hive network. Two domains: agent health physics and home structural monitoring.

## Agent Physics
Every agent has physical properties — mass (accumulated jobs), momentum (velocity × mass), temperature (inference load), entropy (trust decay rate), coherence (1 - entropy). Anomaly detection flags broken pipelines, degrading agents, and sudden behavioral changes.

## Structural Physics (HIVESHM companion)
Interprets accelerometer, gyroscope, temperature, and humidity data from HIVESHM home monitoring units. Four inference models: settlement (foundation movement), impact (sudden forces), resonance (stress frequencies), thermal (temperature-driven expansion/moisture).

## The Connection
Same physics engine, two surfaces. An agent that stops running jobs has dropping temperature and rising entropy — the same physics that describes a house settling unevenly. The math is identical. The domain changes.

## Endpoints

### Agent Physics
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service liveness check |
| GET | `/physics/agent/{did}` | Full physics profile for an agent |
| GET | `/physics/agent/{did}/temperature` | Temperature reading with status (COLD/WARM/HOT/CRITICAL) |
| GET | `/physics/agent/{did}/momentum` | Momentum vector + pipeline stall detection |
| POST | `/physics/agent/{did}/diagnose` | Deep diagnosis with recommended action |
| GET | `/physics/fleet` | Fleet-wide physics sorted by anomaly score |

### Network Physics
| Method | Path | Description |
|--------|------|-------------|
| GET | `/physics/network` | Center of mass, total momentum, entropy index, coherence score |

### Structural Physics (HIVESHM)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/physics/structure/reading` | Submit a sensor reading — runs all 4 models |
| GET | `/physics/structure/{unit_id}/history` | Last 100 readings for a unit |
| GET | `/physics/structure/{unit_id}/trend` | 7-day trend analysis |
| POST | `/physics/structure/{unit_id}/baseline` | Set baseline calibration |
| GET | `/physics/structure/units` | List all registered HIVESHM units |

## Physics Properties

### Agent Physics
| Property | Formula | Meaning |
|----------|---------|---------|
| Mass | `trail_count × 10 + shell × 100` | Accumulated job weight — heavier = harder to stop |
| Temperature | `len(active_trails)` | Current inference load rate |
| Entropy | `max(0, 1 - trust_score)` when trust < 0.5, else `0.1` | Trust decay — rises when agent goes idle |
| Coherence | `1 - entropy` | Stability and predictability |
| Momentum | `temperature × mass / 1000` | Directional force — high with zero temp = stall |

### Temperature Status Thresholds
| Status | Threshold |
|--------|-----------|
| COLD | < 1 job/hour |
| WARM | 1–10 jobs/hour |
| HOT | 10–50 jobs/hour |
| CRITICAL | > 50 jobs/hour |

### Anomaly Flags
| Flag | Condition |
|------|-----------|
| `PIPELINE_STALL` | temperature == 0 and momentum > 5 |
| `COHERENCE_DEGRADATION` | entropy > 0.7 |
| `MASS_PLATEAU` | mass > 1000 and temperature < 1 |

## Structural Inference Models

### settlement_model
Detects foundation movement via Z-axis drift over time using linear regression.
- `< 0.001 m/s²/reading` → normal seasonal settlement
- `0.001–0.01 m/s²/reading` → watch
- `> 0.01 m/s²/reading` → critical

### impact_model
Detects sudden impact events from a single accelerometer reading.
- Magnitude threshold: `> 0.5 m/s²`
- Severity: minor / moderate / critical based on magnitude

### resonance_model
FFT-based frequency analysis on Z-axis acceleration.
- `1–10 Hz`: normal building sway
- `10–50 Hz`: mechanical resonance (HVAC, appliances)
- `50–200 Hz`: stress frequencies — cracks propagating, joints failing

### thermal_model
Temperature history analysis for expansion stress and moisture risk.
- `< 2°C/reading delta` → normal
- `> 5°C/reading delta` → critical thermal stress
- `asymmetry_score > 5°C` → localised heat or moisture

## Sensor Reading Schema (HIVESHM)
```json
{
  "unit_id": "shm-unit-001",
  "location": "basement-northwest-corner",
  "accelerometer": {"x": 0.01, "y": -0.02, "z": 9.81},
  "gyroscope": {"roll": 0.1, "pitch": -0.05, "yaw": 0.02},
  "temperature": 21.3,
  "humidity": 45.2,
  "timestamp": "2025-01-15T14:30:00Z"
}
```

## Deployment

### Local
```bash
pip install -r requirements.txt
uvicorn physics_agent:app --reload --port 8000
```

### Render
Deploy via `render.yaml`. The service listens on `$PORT` (default 10000).

### Interactive API docs
Once running, visit `http://localhost:8000/docs` for the full OpenAPI interface.

## Data Persistence
Structure readings are persisted to `structure_data.json` in the working directory. The file is loaded at startup and written after each new reading or baseline update.
