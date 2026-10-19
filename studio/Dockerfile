FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip \
 && python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && python -m pip install -r studio/requirements-studio.txt -r studio/requirements-extended.txt \
 && python studio/train_export.py
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=6)" || exit 1
CMD ["streamlit","run","studio/app.py","--server.address=0.0.0.0","--server.port=8501","--server.headless=true"]
