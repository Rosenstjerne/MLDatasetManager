FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev]"

CMD ["pytest"]
