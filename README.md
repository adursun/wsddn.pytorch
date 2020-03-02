# WSDDN PyTorch

Implementation of `Weakly Supervised Deep Detection Networks` using the latest version of PyTorch.

*```Bilen, H., & Vedaldi, A. (2016). Weakly supervised deep detection networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2846-2854).```*

## Implementation Differences

- Adam optimizer (instead of SGD)
- Spatial regulariser isn't added

## Experiments

- This implementation is closest to `EB + Box Sc.` case with **L**arge base model, which reported **30.4** mAP in the paper
- Results when `VGG16` is used as base model

| aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv |  mean  |
|------|------|------|------|--------|-----|-----|-----|-------|-----|-------|-----|-------|-------|--------|-------|-------|------|-------|----|--------|
| 41.4 | 46.3 | 22.7 | 24.5 |  13.6  |57.7 |49.9 |31.1 | 7.5   |31.1 | 24.3  |25.9 | 38.7  | 53.5  |  7.2   | 13.9  | 31.1  | 38.6 | 48.3  |39.0|**32.3**|

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
