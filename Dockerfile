FROM python:3.11-slim

# Install system deps (ffmpeg + build tools)
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential git libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src
COPY models/ /app/models

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 