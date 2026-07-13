# Base image
FROM python:3.10-slim

# Set environment variables to prevent python writing pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies needed for computer vision, PyTorch, and debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Build argument for the CUDA/CPU version tag (e.g., cpu, cu118, cu121, cu124)
ARG CUDA_VERSION=cpu

# Install PyTorch, torchvision, and torchaudio with the specified index URL
RUN echo "Installing PyTorch for platform/CUDA tag: ${CUDA_VERSION}..." && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 5001 (used by the Flask app)
EXPOSE 5001

# Command to execute the application
CMD ["python", "app.py"]
