import logging
import random
from collections import defaultdict
from datetime import datetime

import chainercv.transforms as T
import numpy as np
import torch
from tqdm import tqdm
from albumentations import BboxParams, Compose, HorizontalFlip, LongestMaxSize
from albumentations.pytorch.transforms import ToTensor
from chainercv.evaluations import eval_detection_voc
from detectron2.evaluation import PascalVOCDetectionEvaluator
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms

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


def evaluate_detectron2(net, dataloader):
    CLASSES = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    class Detectron2VOCEvaluator(PascalVOCDetectionEvaluator):
        def __init__(self):
            self._dataset_name = "voc_2007_test"
            self._anno_file_template = (
                "/ws/data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations/{}.xml"
            )
            self._image_set_path = (
                "/ws/data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/test.txt"
            )
            self._class_names = CLASSES
            self._is_2007 = True
            self._cpu_device = torch.device("cpu")
            self._logger = logging.getLogger(__name__)
            self._predictions = defaultdict(list)

    evaluator = Detectron2VOCEvaluator()

    print("Evaluation started at", datetime.now())

    with torch.no_grad():

        net.eval()

        # check img_id -> batch or single

        for (img_id, img, boxes, scores, gt_boxes, gt_labels) in dataloader:
            boxes, scores, gt_boxes, gt_labels = (
                boxes.numpy(),
                scores.numpy(),
                gt_boxes.numpy(),
                gt_labels.numpy(),
            )

            batch_imgs, batch_boxes, batch_scores = (
                np2gpu(img, DEVICE),
                np2gpu(boxes, DEVICE),
                np2gpu(scores, DEVICE),
            )

            combined_scores, pred_boxes = net(batch_imgs, batch_boxes, batch_scores)

            for i in range(20):
                region_scores = combined_scores[:, i]

                selected_indices = nms(pred_boxes, region_scores, 0.4)

                resulting_boxes = pred_boxes[selected_indices].cpu().numpy()[:300]
                resulting_scores = region_scores[selected_indices].cpu().numpy()[:300]
                resulting_scores *= np.squeeze(scores[: len(resulting_scores)])

                for j, resulting_box in enumerate(resulting_boxes):
                    evaluator._predictions[i].append(
                        f"{img_id} {resulting_scores[j]:.3f} {resulting_box[0] + 1:.1f} {resulting_box[1] + 1:.1f} {resulting_box[2]:.1f} {resulting_box[3]:.1f}"
                    )

        print("Predictions completed at", datetime.now())

        net.train()

    result = evaluator.evaluate()

    print("Evaluation completed at", datetime.now())
    print(result)


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

            combined_scores, pred_boxes = net(batch_imgs, batch_boxes, batch_scores)

            batch_pred_boxes = []
            batch_pred_scores = []
            batch_pred_labels = []

            for i in range(20):
                region_scores = combined_scores[:, i]
                selected_indices = nms(pred_boxes, region_scores, 0.4)

                batch_pred_boxes.append(pred_boxes[selected_indices].cpu().numpy())
                batch_pred_scores.append(region_scores[selected_indices].cpu().numpy())
                batch_pred_labels.append(
                    np.full(len(selected_indices), i, dtype=np.int32)
                )

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
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep


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
