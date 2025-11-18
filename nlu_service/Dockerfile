# NLU Service - FastAPI service
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including ffmpeg libraries for PyAV
RUN apt-get update && apt-get install -y \
    curl \
    pkg-config \
    build-essential \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy service code
COPY policy.py /app/policy.py

# Expose port
EXPOSE 8001

# Run the service
CMD ["python3", "policy.py"]

