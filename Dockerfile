FROM nvidia/cuda:10.0-devel-ubuntu18.04

WORKDIR /ws

COPY requirements.txt /ws/
COPY install_detectron2.sh /ws/

RUN apt update && apt install -y apt-utils git vim libsm6 libxext6 libxrender-dev python3 python3-dev python3-pip
RUN pip3 install -r requirements.txt

RUN ./install_detectron2.sh
