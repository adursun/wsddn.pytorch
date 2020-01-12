# WSDDN PyTorch

## Requirements

- Docker (19.03.2)
- nvidia-container-toolkit (https://github.com/NVIDIA/nvidia-docker)

## Build Steps

```
./prepare.sh
docker run --rm --gpus all --ipc=host -v `pwd`:/ws -it wsddn.pytorch /bin/bash
```

## Jupyter

```
docker build -f Dockerfile.Jupyter -t wsddn.pytorch:jupyter .
```

## Training Steps

```
cd src
python3 train.py
```

## Evaluation Steps

```
cd src
python3 evaluate.py --path=<state_dict_path>
```
