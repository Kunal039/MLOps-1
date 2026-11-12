FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data/ data/
COPY train_model.py app.py ./
RUN python train_model.py
EXPOSE 5000
# a single worker keeps the in-memory metrics consistent
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
