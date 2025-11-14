# Connectors Service - FastAPI service
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy service code
COPY crm.py /app/crm.py
COPY livechat.py /app/livechat.py
COPY server.py /app/server.py

# Expose port
EXPOSE 8090

# Run the service
CMD ["python3", "server.py"]

