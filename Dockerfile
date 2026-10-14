# Inference service. Build from the repo root:  docker build -t graphprobe-api .
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 REGISTRY_DIR=/app/mlops/registry
# CPU-only PyTorch keeps the image small
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY mlops/requirements-serve.txt mlops/requirements-serve.txt
RUN pip install -r mlops/requirements-serve.txt
COPY scratch_gcn/ scratch_gcn/
COPY mlops/ mlops/
RUN rm -rf mlops/registry_demo
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "mlops.service:app", "--host", "0.0.0.0", "--port", "8000"]
