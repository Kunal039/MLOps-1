"""
Experiment 6: Containerize a machine learning model using Docker
----------------------------------------------------------------
Step 1 - `train`  : train the model that is packaged inside the Docker image
                    (runs during `docker build`, see the RUN line in the Dockerfile).
Step 2 - `predict`: batch / command-line prediction with the packaged model
                    (the container's ENTRYPOINT).
Step 3 - `docker` : write outputs/docker/ = a complete Docker build context
                    (Dockerfile, requirements.txt, .dockerignore, this script and the data).

Usage:
    python docker_ml_model.py                              -> train, predict samples, write the Docker build context
    python docker_ml_model.py train                        -> train and save the model
    python docker_ml_model.py predict                      -> predict the built-in samples
    python docker_ml_model.py predict 5.1 3.5 1.4 0.2      -> predict one flower
    python docker_ml_model.py docker                       -> write outputs/docker/ and print the Docker commands
Inside Docker:
    docker run --rm iris-ml-model:1.0 6.7 3.0 5.2 2.3
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Iris.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "iris_model.joblib"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
SAMPLES = [[5.1, 3.5, 1.4, 0.2], [6.0, 2.7, 4.1, 1.0], [6.7, 3.0, 5.2, 2.3]]


# ---------------------------------------------------------------------------
# Step 1: train
# ---------------------------------------------------------------------------
def train():
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df["Species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling + Logistic Regression wrapped in one pipeline, so the container only needs one file
    model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(classification_report(y_test, pred))
    print(f"Test accuracy: {accuracy:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    (MODEL_DIR / "model_info.json").write_text(
        json.dumps({"model": "LogisticRegression + StandardScaler", "features": FEATURES,
                    "accuracy": round(accuracy, 4), "version": "1.0"}, indent=2)
    )
    print("Model saved to", MODEL_PATH)


# ---------------------------------------------------------------------------
# Step 2: predict
# ---------------------------------------------------------------------------
def load_model():
    if not MODEL_PATH.exists():
        sys.exit("Model not found - run `python docker_ml_model.py train` first.")
    return joblib.load(MODEL_PATH)


def predict(model, rows):
    df = pd.DataFrame(rows, columns=FEATURES)
    labels = model.predict(df)
    confidence = model.predict_proba(df).max(axis=1)
    return list(zip(labels, confidence))


def run_predict(values=None):
    model = load_model()
    samples = [values] if values else SAMPLES
    for features, (label, conf) in zip(samples, predict(model, samples)):
        print(f"{features} -> {label} (confidence {conf:.2f})")


# ---------------------------------------------------------------------------
# Step 3: Docker build context
# ---------------------------------------------------------------------------
DOCKERFILE = """\
# ---------- Experiment 6: Containerize an ML model ----------
# 1. Base image with Python
FROM python:3.12-slim

# 2. Environment settings: no .pyc files, unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1

# 3. Working directory inside the container
WORKDIR /app

# 4. Install dependencies first (this layer is cached when only code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the code and data
COPY data/ data/
COPY docker_ml_model.py ./

# 6. Train the model while building the image -> the model is baked into the image
RUN python docker_ml_model.py train

# 7. Run as a non-root user (security best practice)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# 8. Default command: predict the built-in samples (pass 4 numbers to predict your own flower)
ENTRYPOINT ["python", "docker_ml_model.py", "predict"]
"""

REQUIREMENTS = "pandas==2.2.3\nscikit-learn==1.5.2\njoblib==1.4.2\n"
DOCKERIGNORE = "__pycache__/\n*.pyc\nmodel/\noutputs/\n"

DOCKER_COMMANDS = """\
Docker commands (Docker Desktop must be running):
  1. Build the image (installs dependencies and trains the model inside the image)
       docker build -t iris-ml-model:1.0 "{context}"
  2. List images
       docker images iris-ml-model
  3. Run the container: predicts the built-in sample flowers
       docker run --rm iris-ml-model:1.0
  4. Run the container on your own input (arguments go to "docker_ml_model.py predict")
       docker run --rm iris-ml-model:1.0 6.7 3.0 5.2 2.3
  5. Look inside the image
       docker run --rm --entrypoint ls iris-ml-model:1.0 -l /app /app/model
  6. Remove the image when finished
       docker rmi iris-ml-model:1.0"""


def write_docker_context():
    context = BASE_DIR / "outputs" / "docker"
    if context.exists():
        shutil.rmtree(context)
    (context / "data").mkdir(parents=True)
    (context / "Dockerfile").write_text(DOCKERFILE)
    (context / "requirements.txt").write_text(REQUIREMENTS)
    (context / ".dockerignore").write_text(DOCKERIGNORE)
    shutil.copy2(Path(__file__), context / "docker_ml_model.py")
    shutil.copy2(DATA_PATH, context / "data" / DATA_PATH.name)
    print(f"Docker build context written to {context}")
    print(DOCKER_COMMANDS.format(context=context))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 6 - train / predict the containerized Iris model")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train", help="train and save the model")
    p = sub.add_parser("predict", help="predict the built-in samples or one flower")
    p.add_argument("values", nargs="*", type=float, help="4 measurements: sepal length/width, petal length/width")
    sub.add_parser("docker", help="write outputs/docker/ build context and print the Docker commands")
    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "predict":
        if args.values and len(args.values) != 4:
            parser.error("predict needs exactly 4 numbers")
        run_predict(args.values or None)
    elif args.command == "docker":
        write_docker_context()
    else:
        train()
        print()
        run_predict()
        print()
        write_docker_context()
