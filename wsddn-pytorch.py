import os
import random
import xml.etree.ElementTree as ET
from datetime import datetime

import chainercv.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from chainercv.evaluations import eval_detection_voc
from chainercv.visualizations import vis_bbox
from IPython.display import display
from PIL import Image
from scipy.io import loadmat
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import alexnet
from torchvision.ops import nms, roi_pool

# Some constants
SEED = 61
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SCALES = [480, 576, 688, 864, 1200]
TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Utils
def hflip(img, boxes, gt_boxes=None):
    img = T.flip(img, y_flip=False, x_flip=True)
    boxes = T.flip_bbox(boxes, img[0].shape, y_flip=False, x_flip=True)
    if gt_boxes is None:
        return img, boxes, None
    gt_boxes = T.flip_bbox(gt_boxes, img[0].shape, y_flip=False, x_flip=True)
    return img, boxes, gt_boxes


def scale(img, boxes, max_dim, gt_boxes=None):
    initial_size = img[0].shape
    img = T.scale(img, max_dim, fit_short=False)
    boxes = T.resize_bbox(boxes, initial_size, img[0].shape)
    if gt_boxes is None:
        return img, boxes, None
    gt_boxes = T.resize_bbox(gt_boxes, initial_size, img[0].shape)
    return img, boxes, gt_boxes


def swap_axes(boxes):
    boxes = np.stack((boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]), axis=1)
    return boxes


def np2gpu(arr):
    arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr).to(DEVICE)


class VOCandMCG(Dataset):

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

    def __init__(self, split):
        self.split = split

        # loaded_mat = loadmat(f"/kaggle/input/selective-search-windows/selective_search_data/voc_2007_{self.split}.mat")
        loaded_mat = loadmat(f"data/selective_search_data/voc_2007_{self.split}.mat")
        self.ssw_boxes = loaded_mat["boxes"][0]
        self.ssw_scores = loaded_mat["boxScores"][0]

        # voc_dir = f"/kaggle/input/pascal-voc/voc{self.split}_06-nov-2007/VOCdevkit/VOC2007"
        voc_dir = f"data/VOC{self.split}_06-Nov-2007/VOCdevkit/VOC2007"
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
        target = np.full(20, -1.0, dtype=np.float32)

        for label in gt_labels:
            target[label] = 1.0

        return target

    def _get_annotations(self, i):
        xml = ET.parse(self.annotation_paths[i])

        boxes = []
        labels = []

        for obj in xml.findall("object"):
            if obj.find("difficult").text == "0":
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

    @staticmethod
    def prepare(img, boxes, max_dim=None, xflip=False, gt_boxes=None):
        img = np.asarray(img, dtype=np.float32)  # use numpy array for augmentation
        img = np.transpose(img, (2, 0, 1))  # convert img to CHW

        # convert boxes into (ymin, xmin, ymax, xmax) format
        boxes = swap_axes(boxes)

        if gt_boxes is not None:
            gt_boxes = swap_axes(gt_boxes)

        # scale
        if max_dim:
            img, boxes, gt_boxes = scale(img, boxes, max_dim, gt_boxes)

        # horizontal flip
        if xflip:
            img, boxes, gt_boxes = hflip(img, boxes, gt_boxes)

        # convert boxes back to (xmin, ymin, xmax, ymax) format
        boxes = boxes = swap_axes(boxes)

        if gt_boxes is not None:
            gt_boxes = swap_axes(gt_boxes)

        # convert img from CHW to HWC
        img = Image.fromarray(np.transpose(img, (1, 2, 0)).astype(np.uint8), mode="RGB")
        img = TRANSFORMS(img)  # convert pillow image to normalized tensor

        return img, boxes, gt_boxes

    def __getitem__(self, i):
        img = Image.open(self.img_paths[i]).convert("RGB")  # open Pillow image

        boxes, scores = self.get_boxes_and_scores(i)
        gt_boxes, gt_labels = self._get_annotations(i)

        if self.split == "test":
            return img, boxes, scores, gt_boxes, gt_labels

        img, boxes, _ = self.prepare(
            img, boxes, random.choice(SCALES), random.choice([False, True])
        )
        target = self.get_target(gt_labels)
        return img, boxes, scores, target

    def __len__(self):
        return len(self.ids)


class WSDDN(nn.Module):
    base = alexnet(pretrained=False)

    def __init__(self):
        super().__init__()
        self.base.load_state_dict(torch.load("states/alexnet-owt-4df8aa71.pth"))
        self.features = self.base.features[:-1]
        self.fcs = self.base.classifier[1:-1]
        self.fc_c = nn.Linear(4096, 20)
        self.fc_d = nn.Linear(4096, 20)

    def forward(self, batch_imgs, batch_boxes, batch_scores):
        # assume batch size is 1
        batch_boxes = [batch_boxes[0]]

        out = self.features(batch_imgs)  # [1, 256, 21, 29]
        out = roi_pool(out, batch_boxes, (6, 6), 1.0 / 16)  # [4000, 256, 6, 6]

        out = out.view(-1, 9216)  # [4000, 9216]
        out = self.fcs(out)  # [4000, 4096]

        classification_scores = F.softmax(self.fc_c(out), dim=1)
        detection_scores = F.softmax(self.fc_d(out), dim=0)
        combined_scores = classification_scores * detection_scores

        return combined_scores, batch_boxes[0]


def loss_func(combined_scores, target):
    image_level_scores = torch.sum(combined_scores, dim=0)
    image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
    loss = -torch.sum(torch.log(target * (image_level_scores - 0.5) + 0.50001))
    return loss


if __name__ == "__main__":
    # Set the seed
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Set the hyperparameters
    LR = 1e-5
    WD = 5e-4
    EPOCHS = 40
    OFFSET = 0

    # Create dataset and data loader
    train_ds = VOCandMCG("trainval")
    test_ds = VOCandMCG("test")

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
    test_dl = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=1)

    # Create the network
    if OFFSET == 0:
        net = WSDDN()
    else:
        net = torch.load(f"states/epoch_{OFFSET}.pt")

    net.to(DEVICE)
    net.train()

    # Set loss function and optimizer
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.3)
    scheduler.last_epoch = OFFSET

    # Train the model
    for epoch in range(OFFSET + 1, EPOCHS + 1):

        print(
            "Epoch",
            epoch,
            "started at",
            datetime.now(),
            "with lr =",
            scheduler.get_lr(),
        )

        epoch_loss = 0.0

        for (batch_imgs, batch_boxes, batch_scores, batch_target) in train_dl:
            optimizer.zero_grad()

            batch_imgs, batch_boxes, batch_scores, batch_target = (
                batch_imgs.to(DEVICE),
                batch_boxes.to(DEVICE),
                batch_scores.to(DEVICE),
                batch_target.to(DEVICE),
            )
            combined_scores, _ = net(batch_imgs, batch_boxes, batch_scores)

            loss = loss_func(combined_scores, batch_target[0])
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()

        torch.save(net, f"states/epoch_{epoch}.pt")

        print("Avg loss is", epoch_loss / len(train_ds))

        if epoch % 5 == 0:
            print("Evaluation started at", datetime.now())

            with torch.no_grad():
                net.eval()

                aps = []
                maps = []

                for max_dim in SCALES:

                    for xflip in [True, False]:
                        total_pred_boxes = []
                        total_pred_scores = []
                        total_pred_labels = []
                        total_gt_boxes = []
                        total_gt_labels = []

                        for (img, boxes, scores, gt_boxes, gt_labels) in test_dl:
                            boxes, scores, gt_boxes, gt_labels = (
                                boxes.numpy(),
                                scores.numpy(),
                                gt_boxes.numpy(),
                                gt_labels.numpy(),
                            )
                            p_img, p_boxes, p_gt_boxes = VOCandMCG.prepare(
                                img, boxes, max_dim, xflip, gt_boxes
                            )

                            batch_imgs, batch_boxes, batch_scores, batch_gt_boxes, batch_gt_labels = (
                                np2gpu(p_img),
                                np2gpu(p_boxes),
                                np2gpu(scores),
                                np2gpu(p_gt_boxes),
                                np2gpu(gt_labels),
                            )
                            combined_scores, pred_boxes = net(
                                batch_imgs, batch_boxes, batch_scores
                            )
                            pred_scores, pred_labels = torch.max(combined_scores, dim=1)

                            batch_pred_boxes = []
                            batch_pred_scores = []
                            batch_pred_labels = []

                            for i in range(20):
                                region_scores = combined_scores[:, i]

                                selected_indices = nms(pred_boxes, region_scores, 0.4)

                                batch_pred_boxes.append(
                                    pred_boxes[selected_indices].cpu().numpy()
                                )
                                batch_pred_scores.append(
                                    region_scores[selected_indices].cpu().numpy()
                                )
                                batch_pred_labels.append(
                                    np.full(len(selected_indices), i, dtype=np.int32)
                                )

                            total_pred_boxes.append(
                                np.concatenate(batch_pred_boxes, axis=0)
                            )
                            total_pred_scores.append(
                                np.concatenate(batch_pred_scores, axis=0)
                            )
                            total_pred_labels.append(
                                np.concatenate(batch_pred_labels, axis=0)
                            )
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
                        aps.append(result["ap"])
                        maps.append(result["map"])

                aps = np.stack(aps)
                maps = np.array(maps)

                print("Avg ap:", np.mean(aps, axis=0))
                print("Avg map:", np.mean(maps))

                net.train()

        print("Epoch", epoch, "completed at", datetime.now(), "\n")

        scheduler.step()
