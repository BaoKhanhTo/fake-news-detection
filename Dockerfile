FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps that scikit-learn/numpy wheels still occasionally pull in,
# plus build-essential for any source builds in transitive deps.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Trigger underthesea's first-run model download at build time so cold
# requests at runtime are fast (and don't need network access).
RUN python -c "from underthesea import word_tokenize; word_tokenize('khoi tao', format='text')"

COPY . .

# Run migrations and collect static at container start so the SQLite
# history table exists even on a fresh volume.
EXPOSE 8000
CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py runserver 0.0.0.0:8000"]
