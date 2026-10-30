FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
RUN pip install --no-cache-dir "streamlit>=1.40" "httpx>=0.27"

COPY --chown=10001:10001 dashboard/ ./dashboard/

USER 10001:10001

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).status == 200 else 1)"

CMD ["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
