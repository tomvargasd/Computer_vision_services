#!/bin/bash

# Exit on error
set -e

# Navigate to script's directory
cd "$(dirname "$0")"

echo "=================================================="
echo "          CVVision Deployment Script              "
echo "=================================================="

# Ensure cvvision.db exists as a file so Docker does not mount it as a directory
if [ ! -f "cvvision.db" ]; then
    echo "Creating empty cvvision.db file to prepare for volume mount..."
    touch cvvision.db
fi

# Ensure upload subfolders exist on the host
echo "Ensuring upload directories exist..."
mkdir -p static/uploads/videos
mkdir -p static/uploads/models
mkdir -p static/uploads/captures

# Detect operating system
OS_TYPE=$(uname -s)
echo "System OS detected: $OS_TYPE"

CUDA_TAG="cpu"

if [ "$OS_TYPE" = "Darwin" ]; then
    echo "MacOS detected. Docker on MacOS does not support CUDA GPU passthrough. Defaulting to CPU mode."
elif [ "$OS_TYPE" = "Linux" ]; then
    # Check if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        # Check if there are active NVIDIA GPUs
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l || echo "0")
        if [ "$GPU_COUNT" -gt 0 ]; then
            # Parse CUDA version from nvidia-smi output
            CUDA_VERSION_FULL=$(nvidia-smi | grep -o "CUDA Version: [0-9]*\.[0-9]*" | cut -d':' -f2 | xargs || echo "")
            
            if [ -n "$CUDA_VERSION_FULL" ]; then
                echo "CUDA detected: Version $CUDA_VERSION_FULL"
                CUDA_MAJOR=$(echo "$CUDA_VERSION_FULL" | cut -d'.' -f1)
                CUDA_MINOR=$(echo "$CUDA_VERSION_FULL" | cut -d'.' -f2)
                
                # Match to the best supported PyTorch index URL tag
                if [ "$CUDA_MAJOR" -eq 12 ]; then
                    if [ "$CUDA_MINOR" -ge 4 ]; then
                        CUDA_TAG="cu124"
                    else
                        CUDA_TAG="cu121"
                    fi
                elif [ "$CUDA_MAJOR" -eq 11 ]; then
                    CUDA_TAG="cu118"
                else
                    # General fallback for older/other CUDA versions
                    CUDA_TAG="cu121"
                fi
            else
                echo "Could not parse CUDA version from nvidia-smi. Defaulting to CPU mode."
            fi
        else
            echo "nvidia-smi detected but no NVIDIA GPU devices found. Defaulting to CPU mode."
        fi
    else
        echo "nvidia-smi command not found. Hardware is not CUDA-capable or NVIDIA drivers/toolkit are missing. Defaulting to CPU mode."
    fi
else
    echo "Unknown/unsupported OS '$OS_TYPE'. Defaulting to CPU mode."
fi

# Export variables for docker-compose interpolation
export CUDA_TAG=$CUDA_TAG
echo "Selected PyTorch installation target: $CUDA_TAG"
echo "=================================================="

# Run Docker Compose with conditional file application
if [ "$CUDA_TAG" = "cpu" ]; then
    echo "Starting container in CPU mode..."
    docker compose up --build -d
else
    echo "Starting container in GPU mode (exposing CUDA device reserves)..."
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
fi

echo "=================================================="
echo "Deployment successful! You can monitor the application logs using:"
echo "  docker compose logs -f"
echo "=================================================="
