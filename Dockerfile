FROM ubuntu:focal as pt_to_onnx

ENV DEBIAN_FRONTEND=noninteractive

ARG MODEL_PATH=./model.pt
ADD ${MODEL_PATH} /model.pt

RUN apt-get update \
    && apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6
RUN pip3 install ultralytics
RUN yolo export model=model.pt format=onnx

FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

RUN apt-get update && apt-get install -y wget python3-pip git

WORKDIR /tmp

RUN git clone --recursive https://github.com/microsoft/onnxruntime

WORKDIR onnxruntime

RUN apt-get install -y --no-install-recommends \
    build-essential software-properties-common cmake libopenblas-dev \
	libpython3.6-dev python3-pip python3-dev

RUN ./build.sh --update --config Release --build --build_wheel \
   --use_cuda --cuda_home /usr/local/cuda-10.2 --cudnn_home /usr/lib/aarch64-linux-gnu

#ENV ONNX_WHL=onnxruntime_gpu-1.11.0-cp36-cp36-linux_aarch64.whl
ENV ONNX_WHL=onnxruntime_gpu-1.11.0-any-none-any.whl \
    ONNX_INSTALL=https://nvidia.box.com/shared/static/bfs688apyvor4eo8sf3y1oqtnarwafww.whl
RUN wget ${ONNX_INSTALL} -O ${ONNX_WHL}
RUN python3 -m pip install ${ONNX_WHL}


RUN useradd -m --uid 1000 dockeruser && groupmod --gid 985 video && usermod -a -G video dockeruser
RUN mkdir -p /opt/detect && chown dockeruser:dockeruser /opt/detect -R
COPY --from=pt_to_onnx /model.onnx /opt/detect/
USER dockeruser

CMD ["/bin/bash"]
