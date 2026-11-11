FROM node:22-bookworm-slim AS frontend
WORKDIR /app/viewer
COPY viewer/package.json viewer/package-lock.json ./
RUN npm ci
COPY viewer/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && mkdir -p /app/runtime
COPY imaging/ imaging/
ENV RADREVIEW_DB=/app/runtime/review.sqlite3

FROM base AS api
COPY --from=frontend /app/viewer/dist viewer/dist
COPY fixtures/ fixtures/
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "2", "imaging.api:create_app()"]

FROM base AS worker
COPY requirements-ml.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1 \
    && pip install --no-cache-dir -r requirements-ml.txt
CMD ["python", "-m", "imaging.worker"]
