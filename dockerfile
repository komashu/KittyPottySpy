FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Faster, smaller numpy/scipy stack
ENV OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 \
    GLOG_minloglevel=2 \
    YOLO_CONFIG_DIR=/app/.ultralytics

WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY . /app

# Make dirs expected by the scripts
RUN mkdir -p samples frames crops meta predictions .ultralytics

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh
CMD ["/app/run.sh"]

