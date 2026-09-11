install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	ruff check . || true

format:
	ruff format . || true

run:
	python scripts/run_bot.py --dry-run

backtest:
	python scripts/run_backtest.py --help

docker-build:
	docker build -t trading-goat .

health:
	python scripts/health_check.py
