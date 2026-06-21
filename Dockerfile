FROM python:3.12

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    portaudio19-dev \
    python3-dev \
    alsa-utils \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone repository
RUN git clone https://github.com/RichardX366/tau2-new.git /app
WORKDIR /app

# Install Python dependencies with caching
RUN pip install --no-cache-dir -e .[all]

# Copy the patch file (if it exists)
COPY ./litellm_patch.py /usr/local/lib/python3.12/site-packages/litellm/litellm_core_utils/logging_worker.py

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*

CMD "python"