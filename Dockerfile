FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY bayesian_rag ./bayesian_rag

RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "bayesian_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
