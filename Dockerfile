FROM python:3.11-slim
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH=/app/uci-ml-lib

COPY uci-ml-lib/requirements.txt /tmp/lib-requirements.txt
COPY titanic/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/lib-requirements.txt -r /tmp/requirements.txt

COPY uci-ml-lib/ /app/uci-ml-lib/
COPY titanic/ /app/titanic/

WORKDIR /app/titanic

CMD ["bash", "start.sh"]
