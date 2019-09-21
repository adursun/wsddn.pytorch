FROM nvidia/cuda:10.0-devel-ubuntu18.04

WORKDIR /ws

COPY requirements.txt /ws/

RUN apt update && apt install -y apt-utils git vim python3 python3-dev python3-pip
RUN pip3 install -r requirements.txt
