# WSDDN PyTorch

## Requirements

- Docker (19.03.2)
- nvidia-container-toolkit (https://github.com/NVIDIA/nvidia-docker)

## Build Steps

```
./prepare.sh
docker run --rm --gpus all --ipc=host -v `pwd`:/ws -it wsddn.pytorch /bin/bash
```

## Training Steps

```
python src/train.py
```

## Evaluation Steps

```
python src/evaluate.py --state_path=<state_dict_path>
```
