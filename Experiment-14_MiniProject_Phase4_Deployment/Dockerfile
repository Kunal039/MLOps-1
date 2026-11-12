# Mini project - Phase 4: containerised prediction API
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
# Install the same scikit-learn version the model was trained with, e.g. scikit-learn==1.x.y
RUN pip install --no-cache-dir -r requirements.txt
COPY features.py app.py ./
COPY registry/ registry/
RUN useradd -m appuser
USER appuser
EXPOSE 5000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
