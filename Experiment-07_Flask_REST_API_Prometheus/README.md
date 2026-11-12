# Experiment 7 – REST API with Flask + Prometheus Metrics

**Unit 3 · CO4** | Syllabus: *Create a REST API to serve an ML model using Flask; set up Prometheus to collect metrics; create endpoints to receive input, predict and return results.*

## Aim
To expose an ML model as a web service with input validation, and to publish operational metrics that Prometheus can scrape.

## Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/` | API info and example input |
| GET | `/health` | Health check |
| GET | `/model-info` | Model name, version, accuracy |
| POST | `/predict` | Single `{...}` or batch `{"instances": [...]}` prediction with class probabilities |
| GET | `/metrics` | Prometheus metrics |

## Metrics exposed
`api_requests_total`, `api_request_latency_seconds`, `model_predictions_total{predicted_class}`, `model_prediction_confidence`, `model_test_accuracy`, `model_info`

## Run
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```
In a second terminal:
```bash
python client.py
python -m pytest test_api.py -v
```
To start Prometheus (Docker):
```bash
docker run -p 9090:9090 -v "%cd%/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus
```
Open http://localhost:9090 and query `model_predictions_total`, or `rate(api_requests_total[1m])`.

## Result
5/5 API tests pass. The live run served 55 requests, rejected 1 invalid request with HTTP 400, and Prometheus counters reported 11 setosa, 16 versicolor and 27 virginica predictions.
