FROM coreyhanson/anacuda:base
LABEL maintainer "Corey Ryan Hanson"

#Jupyter's scipy docker image with CUDA runtime added. Code is adapted from their publicly hosted repo https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/ubuntu18.04/10.2/runtime/Dockerfile with slight modifications to keep the user environment the consistent with the Jupyter image.

USER root

ENV NCCL_VERSION 2.5.6

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-$CUDA_PKG_VERSION \
cuda-nvtx-$CUDA_PKG_VERSION \
libcublas10=10.2.2.89-1 \
libnccl2=$NCCL_VERSION-1+cuda10.2 && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*

USER $NB_UID

