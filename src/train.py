import os
import random
from datetime import datetime

import chainercv.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import alexnet
from torchvision.ops import roi_pool

from datasets import VOCandSSW
from utils import (
    evaluate,
    filter_small_boxes,
    hflip,
    np2gpu,
    scale,
    swap_axes,
    unique_boxes,
)

# Some constants
SEED = 61
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SCALES = [480, 576, 688, 864, 1200]


class WSDDN(nn.Module):
    base = alexnet(pretrained=False)

    def __init__(self):
        super().__init__()
        self.base.load_state_dict(torch.load("../states/alexnet-owt-4df8aa71.pth"))
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
    loss = F.binary_cross_entropy(image_level_scores, target, reduction="sum")
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
    train_ds = VOCandSSW("trainval", SCALES)  # len = 5011
    test_ds = VOCandSSW("test", SCALES)  # len = 4952

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
    test_dl = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=1)

    # Create the network
    if OFFSET == 0:
        net = WSDDN()
        print("Training started from the beginning.")
    else:
        net = torch.load(f"../states/epoch_{OFFSET}.pt")
        print(f"Loaded epoch {OFFSET}'s state.")

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

        torch.save(net, f"../states/epoch_{epoch}.pt")

        print("Avg loss is", epoch_loss / len(train_ds))

        if epoch % 10 == 1:
            print("Evaluation started at", datetime.now())
            evaluate(net, SCALES, test_dl)

        print("Epoch", epoch, "completed at", datetime.now(), "\n")

        scheduler.step()
