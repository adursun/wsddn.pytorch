# WSDDN PyTorch

Implementation of `Weakly Supervised Deep Detection Networks` using the latest version of PyTorch.

_`Bilen, H., & Vedaldi, A. (2016). Weakly supervised deep detection networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2846-2854).`_

## Implementation Differences

- Adam optimizer (instead of SGD)
- Spatial regulariser isn't added

## Experiments

- `VGG16` based model is closest to `EB + Box Sc.` case with model **L**, which reported **30.4** mAP in the paper
- `AlexNet` based model is closest to `EB + Box Sc.` case with model **S**, which reported **33.4** mAP in the paper
- Results when `VGG16` is used as base model

| aero | bike | bird | boat | bottle | bus  | car  | cat  | chair | cow  | table | dog  | horse | mbike | person | plant | sheep | sofa | train | tv   | mean     |
| ---- | ---- | ---- | ---- | ------ | ---- | ---- | ---- | ----- | ---- | ----- | ---- | ----- | ----- | ------ | ----- | ----- | ---- | ----- | ---- | -------- |
| 41.4 | 46.3 | 22.7 | 24.5 | 13.6   | 57.7 | 49.9 | 31.1 | 7.5   | 31.1 | 24.3  | 25.9 | 38.7  | 53.5  | 7.2    | 13.9  | 31.1  | 38.6 | 48.3  | 39.0 | **32.3** |

- Results when `AlexNet` is used as base model

| aero | bike | bird | boat | bottle | bus  | car  | cat  | chair | cow  | table | dog  | horse | mbike | person | plant | sheep | sofa | train | tv   | mean     |
| ---- | ---- | ---- | ---- | ------ | ---- | ---- | ---- | ----- | ---- | ----- | ---- | ----- | ----- | ------ | ----- | ----- | ---- | ----- | ---- | -------- |
| 38.1 | 41.5 | 27.1 | 18.6 | 10.3   | 48.8 | 47.6 | 36.8 | 1.6   | 25.9 | 28.5  | 30.4 | 39.7  | 46.8  | 15.1   | 12.4  | 28.3  | 32.4 | 44.2  | 44.8 | **30.9** |

## Requirements

- Docker (19.03.2)
- nvidia-container-toolkit (https://github.com/NVIDIA/nvidia-docker)

## Build Steps

```bash
git clone git@github.com:adursun/wsddn.pytorch.git
cd wsddn.pytorch
./prepare.sh
docker run --rm --gpus all --ipc=host -v `pwd`:/ws -it wsddn.pytorch /bin/bash
```

## Training Steps

```bash
# for VGG based model
python src/train.py --base_net vgg
```

## Evaluation Steps

```bash
# for VGG based model
python src/evaluate.py --base_net vgg --state_path states/vgg_epoch_20.pt
```
