FROM ubuntu:focal as pt_to_onnx

ENV DEBIAN_FRONTEND=noninteractive

ARG MODEL_PATH=./model.pt
ADD ${MODEL_PATH} /model.pt

RUN apt-get update \
    && apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6
RUN pip3 install ultralytics
RUN yolo export model=model.pt format=onnx

#FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3
FROM mcr.microsoft.com/azureml/onnxruntime:v.1.4.0-jetpack4.4-l4t-base-r32.4.3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN useradd -m --uid 1000 dockeruser && groupmod --gid 985 video && usermod -a -G video dockeruser

RUN mkdir -p /opt/detect && chown dockeruser:dockeruser /opt/detect -R
COPY --from=pt_to_onnx /model.onnx /opt/detect/

RUN apt-get update && apt-get install -y python3-pip libprotobuf-dev protobuf-compiler python3-scipy
RUN python3 -m pip install onnx==1.6.0 easydict matplotlib

USER dockeruser

ADD ./src /opt/detect

CMD ["/bin/bash"]
