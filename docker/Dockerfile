FROM nvcr.io/nvidia/cuda:13.1.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_DISABLE_REQUIRE 1

RUN apt update && apt install -y software-properties-common \
    && add-apt-repository universe -y \
    && apt install -y \
    vim \
    sudo \
    git \
    git-lfs \
    python3 \
    python3-pip \
    wget \
    tar \
    pigz \
    curl \
    unzip \
    gdb \
    && rm -rf /var/cache/apt/archives/*

RUN pip3 install flashinfer-bench modal