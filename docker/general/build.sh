#!/usr/bin/env bash

image=mlsys2026-cuda131-general-image
sudo docker build -t ${image} -f Dockerfile .