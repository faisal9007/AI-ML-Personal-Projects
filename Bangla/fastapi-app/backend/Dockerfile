FROM python:3.14-slim

WORKDIR /app

# Copy dependency configuration files
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv pip install --system -r pyproject.toml

# Copy project files
COPY . .

EXPOSE 8000

# Start development application instance
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
