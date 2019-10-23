import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from datasets import VOCandSSW
from network import WSDDN
from utils import evaluate

# Some constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SCALES = [480, 576, 688, 864, 1200]


if __name__ == "__main__":
    # Set the seed
    seed = 61
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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

            loss = WSDDN.calculate_loss(combined_scores, batch_target[0])
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()

        torch.save(net, f"../states/epoch_{epoch}.pt")

        print("Avg loss is", epoch_loss / len(train_ds))

        if epoch % 10 == 1:
            print("Evaluation started at", datetime.now())
            evaluate(net, test_dl)

        print("Epoch", epoch, "completed at", datetime.now(), "\n")

        scheduler.step()
