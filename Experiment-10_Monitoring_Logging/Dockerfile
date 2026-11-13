FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data/ data/
COPY train_model.py monitor.py simulate_traffic.py ./
RUN python train_model.py
EXPOSE 8000
# keep simulating traffic and expose Prometheus metrics on port 8000
CMD ["python", "simulate_traffic.py", "--serve", "--delay", "0.2"]
