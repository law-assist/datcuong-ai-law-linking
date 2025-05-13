FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

# system u
RUN apt-get update

# install Git and Bash
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

# Dev Environment:
    # Python: ✅ 3.10
    # CUDA: ✅ 12.5
    # Linux: ✅ Ubuntu 22.04
    # Git: ✅ 
    # Bash: ✅