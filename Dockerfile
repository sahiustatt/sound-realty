# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + model + demographics (needed at runtime)
COPY app.py app.py
COPY model/ model/
COPY data/zipcode_demographics.csv data/zipcode_demographics.csv

EXPOSE 8000
ENV MODEL_DIR=model
ENV DATA_DIR=data

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

