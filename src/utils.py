import logging
import random
from collections import defaultdict
from datetime import datetime

import chainercv.transforms as T
import numpy as np
import torch
from albumentations import BboxParams, Compose, HorizontalFlip, LongestMaxSize
from albumentations.pytorch.transforms import ToTensor
from chainercv.evaluations import eval_detection_voc
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms

from tqdm import tqdm

# this is duplicate
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_aug(aug):
    return Compose(
        aug, bbox_params=BboxParams(format="pascal_voc", label_fields=["gt_labels"])
    )


def prepare(img, boxes, max_dim=None, xflip=False, gt_boxes=None, gt_labels=None):
    aug = get_aug(
        [
            LongestMaxSize(max_size=max_dim),
            HorizontalFlip(p=float(xflip)),
            ToTensor(
                normalize=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ),
        ]
    )
    augmented = aug(
        image=img, bboxes=boxes, gt_labels=np.full(len(boxes), fill_value=1)
    )
    augmented_gt = aug(image=img, bboxes=gt_boxes, gt_labels=gt_labels)

    img = augmented["image"].numpy().astype(np.float32)
    boxes = np.asarray(augmented["bboxes"]).astype(np.float32)
    gt_boxes = np.asarray(augmented_gt["bboxes"]).astype(np.float32)

    return img, boxes, gt_boxes


def evaluate(net, dataloader):
    """Evaluates network."""
    with torch.no_grad():
        net.eval()

        total_pred_boxes = []
        total_pred_scores = []
        total_pred_labels = []
        total_gt_boxes = []
        total_gt_labels = []

        for (img_id, img, boxes, scores, gt_boxes, gt_labels) in tqdm(
            dataloader, "Evaluation"
        ):
            boxes, scores, gt_boxes, gt_labels = (
                boxes.numpy(),
                scores.numpy(),
                gt_boxes.numpy(),
                gt_labels.numpy(),
            )

            batch_imgs, batch_boxes, batch_scores, batch_gt_boxes, batch_gt_labels = (
                np2gpu(img, DEVICE),
                np2gpu(boxes, DEVICE),
                np2gpu(scores, DEVICE),
                np2gpu(gt_boxes, DEVICE),
                np2gpu(gt_labels, DEVICE),
            )

            combined_scores = net(batch_imgs, batch_boxes, batch_scores)
            pred_boxes = batch_boxes[0]

            batch_pred_boxes = []
            batch_pred_scores = []
            batch_pred_labels = []

            for i in range(20):
                region_scores = combined_scores[:, i]
                score_mask = region_scores > 1e-4

                selected_scores = region_scores[score_mask]
                selected_boxes = pred_boxes[score_mask]
                nms_mask = nms(selected_boxes, selected_scores, 0.4)

                batch_pred_boxes.append(selected_boxes[nms_mask].cpu().numpy())
                batch_pred_scores.append(selected_scores[nms_mask].cpu().numpy())
                batch_pred_labels.append(np.full(len(nms_mask), i, dtype=np.int32))

            total_pred_boxes.append(np.concatenate(batch_pred_boxes, axis=0))
            total_pred_scores.append(np.concatenate(batch_pred_scores, axis=0))
            total_pred_labels.append(np.concatenate(batch_pred_labels, axis=0))
            total_gt_boxes.append(batch_gt_boxes[0].cpu().numpy())
            total_gt_labels.append(batch_gt_labels[0].cpu().numpy())

        result = eval_detection_voc(
            total_pred_boxes,
            total_pred_labels,
            total_pred_scores,
            total_gt_boxes,
            total_gt_labels,
            iou_thresh=0.5,
            use_07_metric=True,
        )

        print("Avg ap:", result["ap"])
        print("Avg map:", result["map"])

        net.train()


def unique_boxes(boxes, scale=1.0):
    """Returns indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def filter_small_boxes(boxes, min_size):
    """Filters out small boxes."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = (w >= min_size) & (h >= min_size)
    return mask


def hflip(img, boxes, gt_boxes=None):
    """Flips image and related boxes horizontally."""
    img = T.flip(img, y_flip=False, x_flip=True, copy=True)
    boxes = T.flip_bbox(boxes, img[0].shape, y_flip=False, x_flip=True)
    if gt_boxes is None:
        return img, boxes, None
    gt_boxes = T.flip_bbox(gt_boxes, img[0].shape, y_flip=False, x_flip=True)
    return img, boxes, gt_boxes


def scale(img, boxes, max_dim, gt_boxes=None):
    """Scales image and related boxes."""
    initial_size = img[0].shape
    scaled_img = T.scale(img, max_dim, fit_short=False)
    boxes = T.resize_bbox(boxes, initial_size, scaled_img[0].shape)
    if gt_boxes is None:
        return scaled_img, boxes, None
    gt_boxes = T.resize_bbox(gt_boxes, initial_size, scaled_img[0].shape)
    return scaled_img, boxes, gt_boxes


def swap_axes(boxes):
    """Swaps x and y axes."""
    boxes = boxes.copy()
    boxes = np.stack((boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]), axis=1)
    return boxes


def np2gpu(arr, device):
    """Creates torch array from numpy one."""
    arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr).to(device)
