# HivePhysics — Build Summary

## Files Created

| File | Lines | Description |
|------|-------|-------------|
| `physics_agent.py` | 1056 | Main FastAPI application |
| `requirements.txt` | 5 | Python dependencies |
| `render.yaml` | 11 | Render.com deployment config |
| `README.md` | 119 | Documentation |
| **Total** | **1191** | |

## physics_agent.py — Structure

### Dependencies
- `fastapi` — web framework and routing
- `uvicorn[standard]` — ASGI server
- `httpx` — async HTTP client for Pulse and HiveForge
- `pydantic` — request/response models
- `numpy` — FFT signal processing for resonance model

### Pydantic Models
| Model | Purpose |
|-------|---------|
| `Vector3` | 3-axis accelerometer / position vector |
| `RotationVector` | Gyroscope roll/pitch/yaw |
| `SensorReading` | HIVESHM sensor reading body |
| `BaselineRequest` | First-install calibration body |
| `DiagnoseResponse` | Agent deep diagnosis response |

### Internal Helper Functions (16)
| Function | Domain |
|----------|--------|
| `_load_structure_data()` | Persistence |
| `_save_structure_data()` | Persistence |
| `_tier_to_shell(tier)` | Agent physics |
| `_temperature_status(temp)` | Agent physics |
| `_build_agent_physics(identity)` | Agent physics |
| `_detect_anomalies(profile)` | Agent physics |
| `_anomaly_score(anomalies)` | Agent physics |
| `_fetch_agent_identity(did)` | Networking |
| `_fetch_census()` | Networking |
| `_settlement_model(readings)` | Structural physics |
| `_impact_model(reading)` | Structural physics |
| `_resonance_model(readings)` | Structural physics (FFT) |
| `_thermal_model(readings)` | Structural physics |
| `_overall_structural_status(...)` | Structural physics |
| `_safe_json(obj)` | Serialisation |

### HTTP Endpoints (13)

#### Meta
| Method | Path |
|--------|------|
| GET | `/health` |

#### Agent Physics
| Method | Path |
|--------|------|
| GET | `/physics/agent/{did}` |
| GET | `/physics/agent/{did}/temperature` |
| GET | `/physics/agent/{did}/momentum` |
| POST | `/physics/agent/{did}/diagnose` |
| GET | `/physics/fleet` |

#### Network Physics
| Method | Path |
|--------|------|
| GET | `/physics/network` |

#### Structural Physics
| Method | Path |
|--------|------|
| POST | `/physics/structure/reading` |
| GET | `/physics/structure/{unit_id}/history` |
| GET | `/physics/structure/{unit_id}/trend` |
| POST | `/physics/structure/{unit_id}/baseline` |
| GET | `/physics/structure/units` |

## Verification

```
cd /home/user/workspace/hive-physics && python -c "import physics_agent; print('Import OK')"
Import OK
```

## Key Design Decisions

1. **In-memory + disk persistence**: Structure readings kept in `deque(maxlen=1000)` per unit; serialised to `structure_data.json` after each write. Loaded at startup.

2. **Graceful error handling**: All `httpx` calls wrapped with `try/except`; `HTTPException` raised with meaningful status codes (502, 504, 404). Global `exception_handler` catches uncaught exceptions.

3. **numpy for signal processing**: `_resonance_model` uses `np.fft.rfft` on Z-axis acceleration. `_settlement_model` uses `np.polyfit` for linear drift regression.

4. **Fleet physics without N+1 calls**: `/physics/fleet` fetches the census once from HiveForge and computes physics from census fields — avoids one Pulse call per agent.

5. **Anomaly scoring**: Each anomaly type has a numeric weight (`PIPELINE_STALL=3.0`, `COHERENCE_DEGRADATION=2.5`, `MASS_PLATEAU=2.0`). Fleet sorted descending by score.

6. **numpy serialisation safety**: `np.floating` and `np.integer` types converted via `.item()` to avoid JSON serialisation errors.

7. **Fully async**: All route handlers and external calls use `async/await`. httpx `AsyncClient` used throughout.
