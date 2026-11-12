# Experiment 6 – Containerize an ML Model with Docker

**Unit 2 · CO4** | Syllabus: *Create a Docker container for the ML model, install dependencies and write a Dockerfile, build and run it locally.*

## Aim
To package a trained ML model, its code and its dependencies into a Docker image, and to run it as a container.

## Theory
- A **Docker image** is a portable snapshot: OS, Python, libraries, code and model. A **container** is a running instance of an image.
- The image solves the "works on my machine" problem.
- **Layer caching:** copy `requirements.txt` first so dependency installation is reused when only code changes.
- **Security:** run as a non-root user.
- **HEALTHCHECK** lets Docker restart an unhealthy service.

## Files
| File | Purpose |
|---|---|
| `train.py` | Trains Logistic Regression + scaler on Iris; runs **inside** `docker build` |
| `predict.py` | Batch/CLI prediction |
| `serve.py` | Small Flask service (`/health`, `/predict`), the container's default command |
| `Dockerfile`, `.dockerignore`, `requirements.txt` | Container definition |
| `docker_commands.sh` | Every Docker command used in this experiment |

## Run (Docker Desktop must be running)
```bash
docker build -t iris-ml-model:1.0 .
docker run --rm iris-ml-model:1.0 python predict.py 5.1 3.5 1.4 0.2
docker run -d --name iris-service -p 5000:5000 iris-ml-model:1.0
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"features\": [6.7, 3.0, 5.2, 2.3]}"
docker logs iris-service
docker stop iris-service
docker rm iris-service
```
To test without Docker: `python train.py && python predict.py && python serve.py`

## Result
Test accuracy is 0.93. `[6.7, 3.0, 5.2, 2.3]` is predicted as **Iris-virginica**.
