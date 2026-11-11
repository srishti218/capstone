FROM node:22-alpine AS dashboard-build
WORKDIR /ui
COPY dashboard/package*.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
ENV VITE_API_URL=/api
RUN npm run build

FROM nginx:alpine AS dashboard
COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=dashboard-build /ui/dist /usr/share/nginx/html

FROM python:3.11-slim AS service-base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-service.txt requirements-graph.txt ./
ARG INSTALL_GNN=false
RUN pip install --no-cache-dir --no-compile -r requirements-service.txt && \
    if [ "$INSTALL_GNN" = "true" ]; then \
      pip install --no-cache-dir --no-compile --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3' && \
      pip install --no-cache-dir --no-compile -r requirements-graph.txt; \
    fi

FROM service-base AS service
COPY src ./src
COPY artifacts ./artifacts
COPY feast ./feast
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
