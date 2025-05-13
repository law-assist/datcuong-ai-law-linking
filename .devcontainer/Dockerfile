FROM nvidia/cuda:12.5.0-base-ubuntu22.04

# system u
RUN apt-get update

# install Git and Bash, Git LFS dependencies
RUN apt-get install -y \
    curl \
    git \
    bash

# install Python 3
RUN apt-get install -y \
    python3.10 \
    python3-pip 

# update PATH
ENV PATH="$PATH:/usr/bin/python3.10"

# install Python dependencies
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip install --no-cache-dir --requirement /tmp/requirements.txt

# Dev Environment:
    # Python: ✅ 3.10
    # Python libraries: Pytorch, FastAPI, Pymongo
    # CUDA: ✅ 12.5
    # Linux: ✅ Ubuntu 22.04
    # Git: ✅ 
    # Bash: ✅