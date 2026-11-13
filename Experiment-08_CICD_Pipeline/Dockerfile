FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas scikit-learn joblib
COPY params.json ./
COPY src/ src/
# model/ and reports/ come from the CI 'train-and-validate' job (validated model only)
COPY model/ model/
COPY reports/ reports/
CMD ["python", "src/predict.py"]
