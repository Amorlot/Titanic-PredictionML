FROM python:3.11-slim
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PYTHONPATH=/app:/app/lib
ENV PYTHONUNBUFFERED=1
ENV LOKY_MAX_CPU_COUNT=8
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x start.sh
CMD ["bash", "start.sh"]
