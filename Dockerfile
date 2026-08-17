FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-download the reranker model so it's baked into the image (no slow download at runtime)
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}

EXPOSE 8000