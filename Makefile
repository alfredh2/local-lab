.PHONY: setup migrate api worker web test lint build

setup:  ## create venv, install backend + frontend deps
	python3 -m venv backend/.venv
	backend/.venv/bin/pip install -e "backend[dev]"
	cd frontend && npm install

migrate:
	cd backend && .venv/bin/alembic upgrade head

api: migrate
	cd backend && .venv/bin/uvicorn locallab.main:app --host $${LOCALLAB_API_HOST:-127.0.0.1} --port $${LOCALLAB_API_PORT:-8321}

worker:
	cd backend && .venv/bin/python -m locallab.worker

web:
	cd frontend && npm run dev

test:
	cd backend && .venv/bin/pytest -q

lint:
	cd frontend && npm run lint && npx tsc --noEmit

build:
	cd frontend && npm run build
