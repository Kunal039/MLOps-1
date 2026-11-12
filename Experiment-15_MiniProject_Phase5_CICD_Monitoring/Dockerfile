# Build from the MLOps-Lab-Experiments folder:
#   docker build -f Experiment-15_MiniProject_Phase5_CICD_Monitoring/Dockerfile -t energy-api:monitored .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir flask pandas numpy scikit-learn joblib prometheus-client gunicorn
COPY Experiment-14_MiniProject_Phase4_Deployment/app.py Experiment-14_MiniProject_Phase4_Deployment/features.py Experiment-14_MiniProject_Phase4_Deployment/
COPY Experiment-14_MiniProject_Phase4_Deployment/registry/ Experiment-14_MiniProject_Phase4_Deployment/registry/
COPY Experiment-15_MiniProject_Phase5_CICD_Monitoring/monitored_app.py Experiment-15_MiniProject_Phase5_CICD_Monitoring/
WORKDIR /app/Experiment-15_MiniProject_Phase5_CICD_Monitoring
EXPOSE 5000
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "monitored_app:app"]
