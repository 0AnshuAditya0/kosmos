FROM node:22-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the serving model into the image. This avoids a hidden Hugging Face
# download and a multi-second cold start when a container is first deployed.
ENV HF_HOME=/opt/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY . .
COPY --from=frontend-build /frontend/dist ./frontend/dist

EXPOSE 7860

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
