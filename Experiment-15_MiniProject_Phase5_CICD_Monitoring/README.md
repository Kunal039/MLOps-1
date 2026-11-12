# Experiment 15 – Mini Project Phase 5: CI/CD, Monitoring & Ethics

> **Mini project:** *Household Electricity Consumption Prediction System* (your proposal: Kunal Mittal, Ansh Kumar, Vishal Kumar).
> Phases 1→5 run in order (Experiments 11 → 15). To run everything at once: `python Experiment-15_MiniProject_Phase5_CICD_Monitoring/run_full_pipeline.py`.

## Aim
To automate the full pipeline with CI/CD, monitor the deployed model in production, detect degradation and trigger retraining, and evaluate privacy, fairness and security.

## Components
| File | Purpose |
|---|---|
| `run_full_pipeline.py` | Runs Phase 1 → 2 → 3 → **quality gate** (≥10% better than baseline) → register → API tests → project tests → monitoring → ethics |
| `.github/workflows/mini-project-cicd.yml` | GitHub Actions: on push, weekly cron retraining and manual trigger; builds and pushes a Docker image |
| `monitored_app.py` | Phase 4 API plus Prometheus metrics, prediction log and a `/actual` feedback endpoint that computes the live 24 h MAE |
| `monitor_and_retrain.py` | Replays ~45 days hour by hour, simulates **concept drift** (EV charging adds 1.2 kW each evening), detects it and writes `retrain_trigger.json` |
| `privacy_ethics_check.py` | Privacy risk, pseudonymization, aggregation, differential-privacy noise, fairness across groups, Model Card, security tests |
| `docker-compose.yml`, `monitoring/` | API + Prometheus (alert rules) + Grafana dashboard |
| `tests/test_project.py` | Data-quality, model-quality and API tests |

## Run
```bash
python run_full_pipeline.py
```
To run the full stack with dashboards:
```bash
docker compose up -d --build
```

## Results
- **Pipeline:** all 9 stages pass.
- **Monitoring:**

  | Period | MAE |
  |---|---|
  | Before drift | 0.053 kW |
  | After drift | 0.219 kW |

  Retraining is triggered automatically once the 24 h MAE exceeds 1.5× the test MAE.
- **Privacy:** minute data shows when the family sleeps (00–05) and is away (11–16). Adding Laplace noise gives this trade-off:

  | ε | MAE |
  |---|---|
  | none | 0.079 |
  | 2 | 0.118 |
  | 0.5 | 0.405 |

- **Fairness:** error is similar for weekday/weekend and for all times of day; no group is more than 25% above average.
- **Security:** all 5 malicious inputs are rejected with HTTP 400.
- **Transparency:** `outputs/MODEL_CARD.md`.
