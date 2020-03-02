# WSDDN PyTorch

Implementation of `Weakly Supervised Deep Detection Networks` using the latest version of PyTorch.

*```Bilen, H., & Vedaldi, A. (2016). Weakly supervised deep detection networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2846-2854).```*

## Implementation Differences
- Adam optimizer (instead of SGD)
- Spatial regulariser isn't added

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
