FROM python:3.12-slim

WORKDIR /app

# Install deps first (Docker caches this layer unless requirements change)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the API needs to run
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000
CMD [\"uvicorn\", \"src.api:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]
