# Mini project - Phase 4: containerised prediction service
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
# Install the same scikit-learn version the model was trained with, e.g. scikit-learn==1.x.y
RUN pip install --no-cache-dir -r requirements.txt
COPY features.py predict.py ./
COPY registry/ registry/
RUN useradd -m appuser
USER appuser
CMD ["python", "predict.py"]
