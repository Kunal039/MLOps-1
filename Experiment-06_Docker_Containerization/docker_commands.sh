#!/usr/bin/env bash
# Experiment 6 - Docker commands (run from this folder, Docker Desktop must be running)

# 1. Build the image (installs dependencies and trains the model inside the image)
docker build -t iris-ml-model:1.0 .

# 2. List images
docker images iris-ml-model

# 3. Run the container: predicts the built-in sample flowers
docker run --rm iris-ml-model:1.0

# 4. Run the container on your own input (arguments go to predict.py)
docker run --rm iris-ml-model:1.0 6.7 3.0 5.2 2.3

# 5. Look inside the image
docker run --rm --entrypoint ls iris-ml-model:1.0 -l /app /app/model

# 6. Remove the image when finished
docker rmi iris-ml-model:1.0
