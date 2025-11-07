# ---------------------------
# Base image
# ---------------------------
FROM python:3.13-slim-bookworm

# System deps: curl for uv installer, and libs for Pillow (jpeg/zlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Python runtime niceties
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---------------------------
# App setup
# ---------------------------
WORKDIR /code

# Copy lockfile + project metadata first to leverage Docker layer caching
COPY pyproject.toml uv.lock /code/

# Install deps (frozen = exactly what’s in uv.lock)
RUN uv sync --frozen

# Copy the application code
# Include helper_lib (API + ML code) and anything else you need at runtime
COPY helper_lib /code/helper_lib
# (Optional) copy train.py if you want to retrain inside the container
# COPY train.py /code/train.py

# If you plan to bake a checkpoint into the image, uncomment this line:
# COPY checkpoints/best.pt /code/checkpoints/best.pt

# ---------------------------
# Runtime config
# ---------------------------
# Default checkpoint path; can be overridden at "docker run" with -e CLASSIFIER_CKPT=/mount/best.pt
ENV CLASSIFIER_CKPT="checkpoints/best.pt"

EXPOSE 8000

# Use uv to run uvicorn
CMD ["uv", "run", "uvicorn", "helper_lib.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ----------------- Test Case ------------------------------
# Use below code to build a docker
# docker build -t sps-genai:latest .

# assuming checkpoints/best.pt exists in your project root, use below code to run
#docker run --rm -p 8000:8000 \
#  -v "$(pwd)/checkpoints:/code/checkpoints:ro" \
#  -e CLASSIFIER_CKPT=/code/checkpoints/best.pt \
#  sps-genai:latest