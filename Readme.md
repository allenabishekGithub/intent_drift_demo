
# Intent Drift Service (FastAPI)

Stores desired intent (constraints) and computes intent drift from posted actuals.

## Quickstart (local)

```bash
pip install -r requirements.txt
uvicorn app:app --reload
# Service on http://127.0.0.1:8000
```

Open docs at: http://127.0.0.1:8000/docs

## Endpoints

- `GET /health` — health check
- `GET /constraints` — list current constraints
- `POST /constraints` — replace all constraints
- `POST /constraints/add` — append a constraint
- `POST /drift` — compute drift for posted actuals

## Example

### 1) Set constraints
```bash
curl -s -X POST http://127.0.0.1:8000/constraints \
  -H 'Content-Type: application/json' \
  -d '{
    "constraints": [
      {"name":"latency_ms","op":"<=","target":10,"weight":2,"norm":10},
      {"name":"loss_pct","op":"<=","target":0.1,"weight":3,"norm":0.1},
      {"name":"throughput_mbps","op":">=","target":1000,"weight":1,"norm":1000}
    ]
  }' | jq .
```

### 2) Compute drift
```bash
curl -s -X POST http://127.0.0.1:8000/drift \
  -H 'Content-Type: application/json' \
  -d '{
    "actuals": {"latency_ms":14, "loss_pct":0.3, "throughput_mbps":800},
    "metric": "l2",
    "theta": 0.5
  }' | jq .
```

### 3) Append a constraint
```bash
curl -s -X POST http://127.0.0.1:8000/constraints/add \
  -H 'Content-Type: application/json' \
  -d '{"name":"jitter_ms","op":"<=","target":1.0,"weight":1,"norm":1.0}' | jq .
```

## Docker

```bash
docker build -t intent-drift-service .
docker run --rm -p 8000:8000 intent-drift-service
```

## Notes

- Constraints are **in-memory**. For persistence, back with a DB (e.g., Redis/Postgres) or mount a volume and serialize.
- The service returns per-feature violations and a boolean `is_drift` based on `theta`.
- Supported metrics: `l2`, `cosine`. For distributional drift, call `IntentDriftEngine.kl_divergence(p,q)` in a custom route.
```
