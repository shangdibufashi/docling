FROM ghcr.io/ai-dock/linux-desktop:pytorch-2.2.1-py3.11-cuda-12.1.0-devel-22.04


ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 curl wget git procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# This will install torch with *only* cpu support
# Remove the --extra-index-url part if you want to install all the gpu requirements
# For more details in the different torch distribution visit https://pytorch.org/.
RUN pip install --no-cache-dir docling torch==2.4 --extra-index-url https://download.pytorch.org/whl/cu120
RUN pip install --no-cache-dir fastapi>=0.115.6 uvicorn>=0.32.1 python-multipart>=0.0.19

ENV HF_HOME=/tmp/
ENV TORCH_HOME=/tmp/

RUN docling-tools models download

ENV DOCLING_ARTIFACTS_PATH=/root/.cache/docling/models

# On container environments, always set a thread budget to avoid undesired thread congestion.
ENV OMP_NUM_THREADS=4

# On container shell:
# > cd /root/
# > python minimal.py

# Running as `docker run -e DOCLING_ARTIFACTS_PATH=/root/.cache/docling/models` will use the
# model weights included in the container image.

COPY main.py /app/src

WORKDIR /app/src

CMD [ "python3", "main.py" ]