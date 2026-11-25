FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-api.txt /tmp/requirements-api.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements-api.txt

COPY src ./src
COPY models ./models
COPY demo ./demo
COPY samples ./samples
COPY pyproject.toml README.md ./

RUN mkdir -p /data
EXPOSE 8000
CMD ["uvicorn", "breplens.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
