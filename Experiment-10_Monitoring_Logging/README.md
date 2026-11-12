# Experiment 10 – Monitoring and Logging with Prometheus & Grafana

**Unit 3 · CO4** | Syllabus: *Use Prometheus and Grafana to monitor the performance of deployed ML models; implement logging to track predictions and model performance in real time.*

## Aim
To monitor a deployed model's traffic, latency, prediction distribution, confidence, input drift and live accuracy, and to log every prediction.

## What is monitored
| Metric | Why |
|---|---|
| `ml_requests_total`, `ml_request_latency_seconds`, `ml_requests_in_progress` | Service health |
| `ml_predictions_total{class}`, `ml_prediction_confidence` | Prediction drift, model uncertainty |
| `ml_feature_value`, `ml_feature_drift_zscore` | **Data drift** against training statistics |
| `ml_live_accuracy` | **Real-time accuracy** from user feedback (`POST /feedback`) |
| `ml_prediction_errors_total` | Bad inputs |

**Logging:** `logs/app.log` is a rotating application log with drift and low-confidence warnings. `logs/predictions.jsonl` holds one JSON record per prediction.

**Alerts** (`monitoring/alert_rules.yml`): DataDriftDetected, LiveAccuracyLow, HighLatency, HighErrorRate.

## Run without Docker
```bash
python train_model.py
python simulate_traffic.py --local
python analyze_logs.py
```

## Run the full stack (Docker Desktop)
```bash
docker compose up -d --build
python simulate_traffic.py --delay 0.2
```
- Grafana: http://localhost:3000 (admin/admin) → dashboard **ML Model Monitoring** (provisioned automatically)
- Prometheus: http://localhost:9090 → Alerts

## Result
| Phase | Accuracy | Notes |
|---|---|---|
| Normal traffic | 96% | |
| Drifted traffic (+2.5 cm petal length) | 73% | Live accuracy metric falls to 0.78; `DATA DRIFT detected on PetalLengthCm` appears in the log |

`outputs/monitoring_report.png` shows confidence, drift and the class mix over time.
