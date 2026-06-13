FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your local source code into the container
COPY src/ /app/src/

# Python can find `src` as a top-level package from /app
ENV PYTHONPATH="/app"