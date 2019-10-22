import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from utils import hflip, np2gpu, prepare, scale, swap_axes, unique_boxes


class VOCandSSW(Dataset):

    CLASS2ID = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "diningtable": 10,
        "dog": 11,
        "horse": 12,
        "motorbike": 13,
        "person": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19,
    }

    def __init__(self, split, scales):
        self.split = split
        self.scales = scales

        loaded_mat = loadmat(f"../data/selective_search_data/voc_2007_{self.split}.mat")
        self.ssw_boxes = loaded_mat["boxes"][0]
        self.ssw_scores = loaded_mat["boxScores"][0]

        voc_dir = f"../data/VOC{self.split}_06-Nov-2007/VOCdevkit/VOC2007"
        self.ids = [
            id_.strip() for id_ in open(f"{voc_dir}/ImageSets/Main/{self.split}.txt")
        ]
        self.img_paths = [f"{voc_dir}/JPEGImages/{id_}.jpg" for id_ in self.ids]
        self.annotation_paths = [f"{voc_dir}/Annotations/{id_}.xml" for id_ in self.ids]

    def get_boxes_and_scores(self, i):
        # (box_count, 4)
        # dtype: float32
        # box format: (y_min, x_min, y_max, x_max)
        boxes = self.ssw_boxes[i].astype(np.float32)

        # box format: (x_min, y_min, x_max, y_max)
        # this can be improved
        boxes = swap_axes(boxes)

        # (box_count, 1)
        # dtype: float64
        scores = self.ssw_scores[i]
        return boxes, scores

    def get_target(self, gt_labels):
        target = np.full(20, 0, dtype=np.float32)

        for label in gt_labels:
            target[label] = 1.0

        return target

    def _get_annotations(self, i):
        xml = ET.parse(self.annotation_paths[i])

        boxes = []
        labels = []

        for obj in xml.findall("object"):
            if obj.find("difficult").text != "1":
                bndbox = obj.find("bndbox")
                boxes.append(
                    [
                        int(bndbox.find(tag).text) - 1
                        for tag in ("xmin", "ymin", "xmax", "ymax")
                    ]
                )
                labels.append(self.CLASS2ID[obj.find("name").text])

        boxes = np.stack(boxes).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)
        return boxes, labels

    def __getitem__(self, i):
        img = Image.open(self.img_paths[i]).convert("RGB")  # open Pillow image

        boxes, scores = self.get_boxes_and_scores(i)
        gt_boxes, gt_labels = self._get_annotations(i)

        if self.split == "test":
            return img, boxes, scores, gt_boxes, gt_labels

        img, boxes, _ = prepare(
            img, boxes, random.choice(self.scales), random.choice([False, True])
        )
        target = self.get_target(gt_labels)
        return img, boxes, scores, target

    def __len__(self):
        return 10  # len(self.ids)
