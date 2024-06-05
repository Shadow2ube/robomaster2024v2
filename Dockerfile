#FROM ubuntu:focal as pt_to_onnx
#
#ENV DEBIAN_FRONTEND=noninteractive
#
#RUN apt-get update \
#    && apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6 git
#RUN pip3 install ultralytics
#RUN pip3 install onnx
#
#ARG MODEL_PATH=./model.pt
#ARG DATA_PATH=./data.yaml
#ARG PT_CONVERTER_PATH=./src/convertv8onnx.py
#ARG VERSION_CONVERTER_PATH=./src/convertonnx15.py
#ADD ${MODEL_PATH} /model.pt
#ADD ${DATA_PATH} /data.yaml
#ADD ${PT_CONVERTER_PATH} /ptconvert.py
#ADD ${VERSION_CONVERTER_PATH} /vconvert.py
#
##RUN yolo export model=model.pt format=onnx simplify=true int8=true opset=15
#RUN python3 /ptconvert.py /model.pt /pre_final.onnx
#RUN python3 /vconvert.py /pre_final.onnx /model.onnx

FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

RUN apt-get update && apt-get install -y wget python3-pip git

WORKDIR /tmp

RUN apt-get install -y --no-install-recommends \
    build-essential software-properties-common libopenblas-dev \
	libpython3.6-dev python3-pip python3-dev

ENV DEBIAN_FRONTEND=noninteractive CMAKE_VERSION=3.18.0
RUN apt-get update && apt-get install -y cmake make wget libssl-dev openssl qt5-default gcc g++ \
    && mkdir /tmp/cmake-build \
    && cd /tmp/cmake-build \
    && wget -c --show-progress https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz \
    && tar xvf cmake-${CMAKE_VERSION}.tar.gz \
    && mkdir ${CMAKE_VERSION}-build \
    && cd ${CMAKE_VERSION}-build \
    && cmake -DBUILD_QtDialog=ON -DQT_QMAKE_EXECUTABLE=/usr/lib/qt5/bin/qmake ../cmake-${CMAKE_VERSION} \
    && make -j $(nproc) \
    && make install

ENV ONNXRUNTIME_VERSION=1.9.1
RUN git clone --branch v${ONNXRUNTIME_VERSION} --recursive https://github.com/microsoft/onnxruntime

WORKDIR onnxruntime

RUN ./build.sh --update --config Release --build --build_wheel \
   --use_cuda --cuda_home /usr/local/cuda-10.2 --cudnn_home /usr/lib/aarch64-linux-gnu

ENV ONNXRUNTIME_WHL=/tmp/onnxruntime/build/Linux/Release/dist/onnxruntime_gpu-${ONNXRUNTIME_VERSION}-cp36-cp36m-linux_aarch64.whl
RUN python3 -m pip install ${ONNXRUNTIME_WHL}

RUN useradd -m --uid 1000 dockeruser && groupmod --gid 985 video && usermod -a -G video dockeruser
RUN mkdir -p /opt/detect && chown dockeruser:dockeruser /opt/detect -R

ARG MODEL_PATH=./models/model.onnx
ADD ${MODEL_PATH} /model.onnx
USER dockeruser

CMD ["/bin/bash"]
