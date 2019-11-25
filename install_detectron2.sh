#!/bin/bash

pip3 install 'git+https://github.com/facebookresearch/fvcore'
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip3 install .
