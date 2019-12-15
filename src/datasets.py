import random
import xml.etree.ElementTree as ET

import cv2 as cv
import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

from utils import TRANSFORMS, filter_small_boxes, prepare, swap_axes


class VocAndEb(Dataset):

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
        assert split in ["trainval", "test"], "`split` should be in [trainval, test]"

        self.split = split
        self.scales = scales

        loaded_mat = loadmat(f"../data/EdgeBoxesVOC2007{self.split}.mat")
        self.eb_boxes = loaded_mat["boxes"][0]
        self.eb_scores = loaded_mat["boxScores"][0]
        self.ids = [str(id_[0]) for id_ in loaded_mat["images"][0]]

        voc_dir = f"../data/VOC{self.split}_06-Nov-2007/VOCdevkit/VOC2007"
        self.img_paths = [f"{voc_dir}/JPEGImages/{id_}.jpg" for id_ in self.ids]
        self.annotation_paths = [f"{voc_dir}/Annotations/{id_}.xml" for id_ in self.ids]

    def get_boxes_and_scores(self, i):
        # (box_count, 4)
        # dtype: float32
        # box format: (y_min, x_min, y_max, x_max)
        boxes = self.eb_boxes[i].astype(np.float32)

        # box format: (x_min, y_min, x_max, y_max)
        # this can be improved
        boxes = swap_axes(boxes)
        mask = filter_small_boxes(boxes, 20)

        # (box_count, 1)
        # dtype: float64
        scores = self.eb_scores[i]
        return boxes[mask], scores[mask]

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
                        int(bndbox.find(tag).text)
                        for tag in ("xmin", "ymin", "xmax", "ymax")
                    ]
                )
                boxes[-1][0] -= 1
                boxes[-1][1] -= 1
                labels.append(self.CLASS2ID[obj.find("name").text])

        boxes = np.stack(boxes).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)
        return boxes, labels

    def __getitem__(self, i):
        img = cv.imread(self.img_paths[i])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        boxes, scores = self.get_boxes_and_scores(i)
        gt_boxes, gt_labels = self._get_annotations(i)

        if self.split == "trainval":
            img, boxes, gt_boxes = prepare(
                img,
                boxes,
                random.choice(self.scales),
                random.choice([False, True]),
                gt_boxes,
                gt_labels,
            )
            target = self.get_target(gt_labels)
            return self.ids[i], img, boxes, scores, target
        elif self.split == "test":
            img_id = self.ids[i]
            orig_img = TRANSFORMS(img)
            scaled_imgs = []
            scaled_boxes = []

            for xflip in [True, False]:
                for max_dim in self.scales:
                    scaled_img, tmp_scaled_boxes, _ = prepare(
                        img,
                        boxes,
                        max_dim=max_dim,
                        xflip=xflip,
                        gt_boxes=[],
                        gt_labels=[],
                    )
                    scaled_imgs.append(scaled_img)
                    scaled_boxes.append(tmp_scaled_boxes)

            return (
                img_id,
                orig_img,
                boxes,
                scaled_imgs,
                scaled_boxes,
                scores,
                gt_boxes,
                gt_labels,
            )

    def __len__(self):
        return len(self.ids)
