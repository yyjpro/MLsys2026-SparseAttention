FROM nvcr.io/nvidia/cuda:13.1.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_DISABLE_REQUIRE 1

ENV TZ Asia/Hong_Kong
RUN apt install --reinstall tzdata && apt clean && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone && rm -rf /var/cache/apt/archives/*

RUN apt update && apt install -y software-properties-common \
    && add-apt-repository universe -y \
    && apt install -y \
    vim \
    sudo \
    git \
    python3 \
    python3-pip \
    wget \
    tar \
    pigz \
    curl \
    unzip \
    gdb \
    && rm -rf /var/cache/apt/archives/*