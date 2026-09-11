FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd -m -u 10001 goat && chown goat:goat /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p logs && chown -R goat:goat /app
USER goat
EXPOSE 5000
HEALTHCHECK --interval=60s --timeout=10s --retries=3 CMD python scripts/health_check.py || exit 1
CMD ["python", "scripts/run_bot.py", "--dry-run"]
