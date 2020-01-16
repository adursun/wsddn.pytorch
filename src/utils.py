import logging
import os
import random
from collections import defaultdict
from datetime import datetime

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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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

        for (
            img_id,
            img,  # is it necessary
            boxes,
            scaled_imgs,
            scaled_boxes,
            scores,
            gt_boxes,
            gt_labels,
        ) in tqdm(dataloader, "Evaluation"):

            combined_scores = torch.zeros(len(boxes), 20, dtype=torch.float32)
            batch_scores = np2gpu(scores.numpy(), DEVICE)

            for i, scaled_img in enumerate(scaled_imgs):
                scaled_img = scaled_img.numpy()
                tmp_scaled_boxes = scaled_boxes[i].numpy()

                batch_imgs = np2gpu(scaled_img, DEVICE)
                batch_boxes = np2gpu(tmp_scaled_boxes, DEVICE)

                tmp_combined_scores = net(batch_imgs, batch_boxes, batch_scores)
                combined_scores += tmp_combined_scores.cpu()

            combined_scores /= 10

            gt_boxes = gt_boxes.numpy()
            gt_labels = gt_labels.numpy()

            batch_gt_boxes = np2gpu(gt_boxes, DEVICE)
            batch_gt_labels = np2gpu(gt_labels, DEVICE)

            batch_pred_boxes = []
            batch_pred_scores = []
            batch_pred_labels = []

            for i in range(20):
                region_scores = combined_scores[:, i]
                score_mask = region_scores > 0

                selected_scores = region_scores[score_mask]
                selected_boxes = boxes[score_mask]

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

        tqdm.write(f"Avg AP: {result['ap']}")
        tqdm.write(f"Avg mAP: {result['map']}")

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


def swap_axes(boxes):
    """Swaps x and y axes."""
    boxes = boxes.copy()
    boxes = np.stack((boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]), axis=1)
    return boxes


def np2gpu(arr, device):
    """Creates torch array from numpy one."""
    arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr).to(device)
