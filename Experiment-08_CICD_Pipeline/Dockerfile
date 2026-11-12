FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir flask pandas scikit-learn joblib gunicorn
COPY app.py params.json ./
COPY src/ src/
# model/ and reports/ come from the CI 'train-and-validate' job (validated model only)
COPY model/ model/
COPY reports/ reports/
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
