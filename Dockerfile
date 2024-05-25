FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

ENV DEBIAN_FRONTEND=noninteractive CMAKE_VERSION=3.18.0

RUN apt-get update && apt-get install -y wget python3-pip git

WORKDIR /tmp

RUN apt-get install -y libfreetype6-dev

RUN git clone https://github.com/ultralytics/yolov5
RUN cd /tmp/yolov5 \
    && sed -i \
      -e 's/torch/# torch/g' \
      -e 's/gitpython>=3.1.30/gitpython>=3.1.20/g' \
      -e 's/numpy>=1.23.5/numpy>=1.19.5/g' \
      -e 's/pillow>=10.3.0/pillow>=8.4.0/g' \
      requirements.txt \
    && pip3 install -r requirements.txt

RUN apt-get install -y libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev

ENV TORCH_WHEEL=torch-1.10.0-cp36-cp36m-linux_aarch64.whl
RUN wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O ${TORCH_WHEEL} && pip3 install ${TORCH_WHEEL}

RUN git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN cd torchvision  && python3 setup.py install
RUN cd /tmp/ultralytics && pip3 install .

#RUN apt-get install -y --no-install-recommends \
#    build-essential software-properties-common libopenblas-dev \
#	libpython3.6-dev python3-pip python3-dev

#RUN useradd -m --uid 1000 dockeruser && groupmod --gid 985 video && usermod -a -G video dockeruser
#RUN mkdir -p /opt/detect && chown dockeruser:dockeruser /opt/detect -R
#USER dockeruser

CMD ["/bin/bash"]
