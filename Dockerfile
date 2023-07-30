FROM python:3.11.4-slim

RUN apt-get update && \
    apt-get install -y git gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /ai

COPY requirements requirements

RUN pip install -r requirements/base.txt

COPY .env.celery .env.celery

COPY app app

ENV MOCKING=True

ENTRYPOINT ["celery", "-A", "app.worker", "worker"]

CMD ["--concurrency=2", "-Ofair"]