# syntax=docker/dockerfile:1
FROM python:3.11.7-slim
LABEL fly_launch_runtime="flask"
WORKDIR /code
ENV UV_SYSTEM_PYTHON=1
COPY pyproject.toml uv.lock ./
RUN pip install uv --no-cache-dir && uv sync --no-dev --frozen
COPY . .
EXPOSE 8080
CMD ["uv", "run", "--no-sync", "python", "app.py"]