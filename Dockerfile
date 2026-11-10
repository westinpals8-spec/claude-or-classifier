# Feedback triage API (CPU-only). Build: docker build -t feedback-triage .
# Run:   docker run -p 8000:8000 -e TRIAGE_API_KEY=... [-e ANTHROPIC_API_KEY=...] [-e TRIAGE_QUANTIZE=1] feedback-triage
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt
COPY src/ src/
COPY models/ models/
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
