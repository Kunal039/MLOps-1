# Experiment 14 – Mini Project Phase 4: Versioning & Deployment

> **Mini project:** *Household Electricity Consumption Prediction System* (your proposal: Kunal Mittal, Ansh Kumar, Vishal Kumar).
> Phases 1→5 run in order (Experiments 11 → 15). To run everything at once: `python Experiment-15_MiniProject_Phase5_CICD_Monitoring/run_full_pipeline.py`.

## Aim
To version the selected model and deploy it as a containerized REST API.

## Components
| File | Purpose |
|---|---|
| `model_registry.py` | Simple model registry. `registry/v1/` holds the model, metadata and SHA-256 checksum; `latest.json` points at the version in use. Supports `--list` and `--rollback v1`. |
| `features.py` | Builds the **same features as Phase 2** from raw recent history. This avoids training-serving skew. |
| `app.py` | Flask API: `/health`, `/model-info`, `/predict` (next hour), `/forecast` (next N hours, recursive) |
| `test_api.py` | 6 tests, including one that API features **exactly equal** the Phase 2 features |
| `Dockerfile` | gunicorn, non-root user, healthcheck |

## Run
```bash
python model_registry.py
python -m pytest test_api.py -v
python app.py
```
With Docker:
```bash
docker build -t household-energy-api:1.0 .
docker run -p 5000:5000 household-energy-api:1.0
```

## Example request
```json
POST /predict
{"timestamp": "2009-06-12 09:00:00",
 "recent_consumption": [ ...168 hourly kW values... ],
 "voltage_last_hour": 239.4, "reactive_power_last_hour": 0.15, "sub_metering_3_last_hour": 7}
-> {"predicted_kw": 0.516, "model_version": "v1"}
```

**Cloud deployment:** push the image to AWS ECR, GCP Artifact Registry or Azure ACR, then run it on ECS/App Runner, Cloud Run or Container Apps.

For DVC data/model versioning, see Experiment 9 (`dvc add`, `dvc.yaml`).
