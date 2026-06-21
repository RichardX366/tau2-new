FROM python:3.12
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir uv
RUN uv pip install --no-cache-dir .[all]
COPY ./litellm_patch.py /usr/local/lib/python3.12/site-packages/litellm/litellm_core_utils/logging_worker.py