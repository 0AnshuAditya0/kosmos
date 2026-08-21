FROM python:3.11-slim

WORKDIR /app

# 1. Memory fragmentation & thread-capping env vars for strict 512MB RAM containers
ENV PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# 2. Install libgomp1 (required for OpenMP by FAISS and ONNX Runtime on debian-slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# 3. Exec form CMD with explicit single worker
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]