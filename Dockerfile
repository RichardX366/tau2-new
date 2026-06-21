# ============================================
# BUILDER STAGE - Contains compilers & build tools
# ============================================
FROM python:3.12 AS builder

# Install build dependencies (compilers, headers, etc.)
RUN apt-get update && \
    apt-get install -y \
    portaudio19-dev \
    python3-dev \
    alsa-utils \
    libasound2-dev \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Clone repository (shallow clone to save space)
RUN git clone --depth 1 https://github.com/RichardX366/tau2-new.git /app
WORKDIR /app

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir -e .[all]

# Apply the patch inside the virtual environment's site-packages
COPY ./litellm_patch.py /opt/venv/lib/python3.12/site-packages/litellm/litellm_core_utils/logging_worker.py

# ============================================
# PRODUCTION STAGE - Runtime only (no compilers)
# ============================================
FROM python:3.12-slim

# Install ONLY runtime dependencies (no -dev packages, no compilers)
RUN wget -qO /usr/local/bin/ttyd https://github.com/tsl0922/ttyd/releases/latest/download/ttyd.x86_64 && \
    chmod a+x /usr/local/bin/ttyd

RUN apt-get update && \
    apt-get install -y \
    libasound2 \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY --from=builder /app /app

# Set working directory and PATH
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH"

# Optional: Remove pip to save space (uncomment if you don't need it)
# RUN /opt/venv/bin/python -m pip uninstall -y pip setuptools wheel

# Default command
CMD ["ttyd", "bash"]