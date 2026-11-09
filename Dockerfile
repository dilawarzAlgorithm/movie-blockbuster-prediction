FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-install-project

COPY data/ ./data/
COPY train.py predict.py serve.py ./

RUN uv run --no-project python train.py

EXPOSE 9696

CMD ["uv", "run", "--no-project", "python", "serve.py"]
