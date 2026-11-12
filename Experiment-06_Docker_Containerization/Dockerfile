# ---------- Experiment 6: Containerize an ML model ----------
# 1. Base image with Python
FROM python:3.12-slim

# 2. Environment settings: no .pyc files, unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Working directory inside the container
WORKDIR /app

# 4. Install dependencies first (this layer is cached when only code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the code and data
COPY data/ data/
COPY train.py predict.py serve.py ./

# 6. Train the model while building the image -> the model is baked into the image
RUN python train.py

# 7. Run as a non-root user (security best practice)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# 8. Document the port and the default command (start the prediction service)
EXPOSE 5000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"
CMD ["python", "serve.py"]
