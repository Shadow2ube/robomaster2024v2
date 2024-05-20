FROM ubuntu:focal as pt_to_onnx

ENV DEBIAN_FRONTEND=noninteractive

ARG MODEL_PATH=./model.pt
ADD ${MODEL_PATH} /model.pt

RUN apt-get update \
    && apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6
RUN pip3 install ultralytics
RUN yolo export model=model.pt format=onnx

FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y locales lsb-release
ENV DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales

RUN apt-get install -y curl

RUN pip3 install onnx onnxruntime-gpu

RUN mkdir -p /opt/detect

COPY --from=pt_to_onnx /model.onnx /opt/detect/

RUN useradd -m --uid 1000 dockeruser \
    && chown dockeruser:dockeruser /opt/detect -R  \
    && groupmod --gid 985 video \
    && usermod -a -G video dockeruser

RUN apt-get install -y cmake make gcc g++

WORKDIR /tmp
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-aarch64-1.17.3.tgz
RUN tar -xvf onnxruntime-linux-aarch64-1.17.3.tgz
RUN mv onnxruntime-linux-aarch64-1.17.3 /opt/onnxruntime

ADD ./ /opt/detect


CMD ["bash"]
