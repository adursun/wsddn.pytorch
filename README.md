# WSDDN PyTorch

## Requirements

- Docker (19.03.2)
- nvidia-container-toolkit (https://github.com/NVIDIA/nvidia-docker)

## Build Steps

```
./prepare.sh
docker run --gpus all -v `pwd`:/ws -it wsddn.pytorch /bin/bash
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
