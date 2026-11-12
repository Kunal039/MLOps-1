#!/usr/bin/env bash
# Experiment 6 - Docker commands (run from this folder, Docker Desktop must be running)

# 1. Build the image
docker build -t iris-ml-model:1.0 .

# 2. List images
docker images iris-ml-model

# 3. Run a one-off batch prediction (overrides the default command)
docker run --rm iris-ml-model:1.0 python predict.py 5.1 3.5 1.4 0.2

# 4. Run the model as a service in the background
docker run -d --name iris-service -p 5000:5000 iris-ml-model:1.0

# 5. Check it is running and test it
docker ps
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"features": [6.7, 3.0, 5.2, 2.3]}'

# 6. View logs, then stop and remove the container
docker logs iris-service
docker stop iris-service && docker rm iris-service
