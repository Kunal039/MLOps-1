FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir flask pandas numpy scikit-learn joblib prometheus-client gunicorn
COPY generate_data.py app.py ./
COPY model/ model/
EXPOSE 5000
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
