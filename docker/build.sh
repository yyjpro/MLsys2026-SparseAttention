#!/usr/bin/env bash

image=cuda131-general-image
sudo docker build -t ${image} -f Dockerfile .