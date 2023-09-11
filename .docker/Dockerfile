ARG CUDA_VERSION=11.7.1
ARG CUDNN_VERSION=cudnn8
ARG UBUNTU_VERSION=22.04

# Base image
FROM nvidia/cuda:${CUDA_VERSION}-${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base

# ENV- Environment Variables
ENV HOME_DIR=/root/
ENV WS_DIR=${HOME_DIR}/workspaces
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Set SHELL to bash
SHELL ["/bin/bash", "-c"]

# APT- Install apt dependencies
# A common list that i use for most projects, includes gl libraries required for GUI apps
RUN apt-get update -y \
    && DEBIAN_FRONTEND=noninteractive \
    apt-get install -q -y --no-install-recommends \
    build-essential \
    curl \
    cmake \
    dirmngr \
    gnupg2 \
    git \
    iputils-ping \
    nano \
    net-tools \
    python3-dev \
    python3-pip \
    python3-wheel \
    tree \
    unzip \
    wget \
    # GUI dependencies
    python3-opengl \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd-dev  \
    libglu1-mesa \
    libsm6 \
    libxi6 \
    libxrandr2 \
    libxt6 \
    vulkan-tools \
    # Remove apt cache to not clutter image layers
    && rm -rf /var/lib/apt/lists/* \
    # Add convenience aliases to bashrc
    && echo "alias python=python3" >> /root/.bashrc\
    && echo "alias pip=pip3" >> /root/.bashrc


## NOTE: Separating COPY/RUN commands into several layers saves time in rebuilding during development
# During deployment/distribution: To optimize image size, they can be combined into one layer

# Python dependencies from requirements.txt
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt


CMD ["/bin/bash"]
