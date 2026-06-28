FROM python:3.11-slim

ENV POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PATH="/root/.local/bin:$PATH"

RUN pip install --no-cache-dir "poetry" \
    && useradd -m -u 1000 user

WORKDIR /app

COPY --chown=user pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root

COPY --chown=user . .
RUN chown -R user:user /app

USER user
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

