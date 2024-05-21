FROM ubuntu:focal as pt_to_onnx

ENV DEBIAN_FRONTEND=noninteractive

ARG MODEL_PATH=./model.pt
ADD ${MODEL_PATH} /model.pt

RUN apt-get update \
    && apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6
RUN pip3 install ultralytics
RUN yolo export model=model.pt format=onnx

FROM ubuntu:focal as cmake_build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y cmake make wget libssl-dev openssl qt5-default gcc g++ \
    && mkdir /tmp/cmake-build
WORKDIR /tmp/cmake-build
RUN wget -c --show-progress https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.tar.gz \
    && tar xvf cmake-3.19.1.tar.gz \
    && mkdir cmake-3.19.1-build
WORKDIR cmake-3.19.1-build
RUN cmake -DBUILD_QtDialog=ON -DQT_QMAKE_EXECUTABLE=/usr/lib/qt5/bin/qmake ../cmake-3.19.1 \
    && make -j $(nproc)

RUN make install

FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

RUN apt-get update && apt-get install -y wget python3-pip git

WORKDIR /tmp

RUN git clone --branch v1.11.0 --recursive https://github.com/microsoft/onnxruntime

RUN apt-get install -y --no-install-recommends \
    build-essential software-properties-common libopenblas-dev \
	libpython3.6-dev python3-pip python3-dev

COPY --from=cmake_build /usr/local/share/cmake-3.19 /usr/local/share/cmake-3.19
COPY --from=cmake_build /usr/local/bin/cmake /usr/local/bin/cmake

WORKDIR onnxruntime

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
