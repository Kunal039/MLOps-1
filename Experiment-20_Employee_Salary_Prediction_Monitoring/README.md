# Experiment 20 – Employee Salary Prediction & Monitoring System using MLOps Tools

**Sample project · CO3/CO4/CO5** | Tools: **MLflow**, Flask, **Prometheus**, **Grafana**, Docker Compose

## Aim
To build a salary prediction model with experiment tracking and a model registry, deploy it with monitoring and logging, detect drift, and audit it for bias.

## Workflow
| Step | Script | What happens |
|---|---|---|
| 1. Data | `generate_data.py` | 3,000 employees (role, education, city, company size, experience, skills, certifications, rating) and a drifted new batch of 800 |
| 2. Track & register | `train_mlflow.py` | 5 algorithms logged to **MLflow** (params, metrics, model). The best is registered as `employee-salary-model` with alias **production** |
| 3. Serve & monitor | `app.py` | Flask API with validation, a prediction interval, Prometheus metrics and a JSON prediction log |
| 4. Drift monitoring | `monitor_drift.py` | **PSI** per feature, prediction drift and performance on new labelled data, ending in a retrain decision |
| 5. Responsible AI | `fairness_audit.py` | Gender pay-gap analysis, error parity, proxy-leakage check. Results are logged to MLflow |
| Infra | `docker-compose.yml`, `monitoring/` | API + MLflow UI + Prometheus (alerts) + Grafana dashboard |

## Run
```bash
pip install -r requirements.txt
python generate_data.py
python generate_data.py --new-batch
python train_mlflow.py
python monitor_drift.py
python fairness_audit.py
python app.py
```
To see the experiments, run `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001` and open http://127.0.0.1:5001.

To start the full stack: `docker compose up -d --build` (Grafana at http://localhost:3000, login admin/admin).

## Results
| Model | Test RMSE (LPA) | R² |
|---|---|---|
| **Gradient Boosting** (production) | **1.89** | **0.927** |
| Ridge / Linear | 2.16 | 0.904 |
| Random Forest | 2.36 | 0.886 |
| Decision Tree | 3.11 | 0.801 |

- **Drift:** experience PSI 0.38 (significant), job role 0.19 (moderate). On new data RMSE rises from 1.89 to 3.88 (+105%), so **retraining is required**.
- **Fairness:** the data has a 6.5% adjusted gender pay gap. The model, which never sees gender, predicts only a 0.8% gap. Gender cannot be inferred from the inputs (proxy AUC 0.51) and MAE is equal across genders. **Verdict: PASS.**
