import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import VocAndEb
from network import WSDDN
from utils import BASE_DIR, evaluate

# Some constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SCALES = [480, 576, 688, 864, 1200]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WSDDN model")
    parser.add_argument("--seed", type=int, default=61, help="Seed to use")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Epoch count")
    parser.add_argument("--offset", type=int, default=0, help="Offset count")
    parser.add_argument("--eval_period", type=int, default=10, help="Evaluation period")
    parser.add_argument(
        "--state_period", type=int, default=5, help="State saving period"
    )
    args = parser.parse_args()

    # Set the seed
    SEED = args.seed
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Set the hyperparameters
    LR = args.lr
    WD = args.wd

    EPOCHS = args.epochs
    OFFSET = args.offset

    EVAL_PERIOD = args.eval_period
    STATE_PERIOD = args.state_period

    # Create dataset and data loader
    train_ds = VocAndEb("trainval", SCALES)  # len = 5011
    test_ds = VocAndEb("test", SCALES)  # len = 4952

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=4)

    # Create the network
    net = WSDDN()

    if OFFSET != 0:
        state_path = os.path.join(BASE_DIR, "states", f"epoch_{OFFSET}.pt")
        net.load_state_dict(torch.load(state_path))
        print(f"Loaded epoch {OFFSET}'s state.")

    net.to(DEVICE)
    net.train()

    # Set loss function and optimizer
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    scheduler.last_epoch = OFFSET

    # Train the model
    for epoch in tqdm(range(OFFSET + 1, EPOCHS + 1), "Total"):

        epoch_loss = 0.0

        for (
            batch_img_ids,
            batch_imgs,
            batch_boxes,
            batch_scores,
            batch_target,
        ) in tqdm(train_dl, f"Epoch {epoch}"):
            optimizer.zero_grad()

            batch_imgs, batch_boxes, batch_scores, batch_target = (
                batch_imgs.to(DEVICE),
                batch_boxes.to(DEVICE),
                batch_scores.to(DEVICE),
                batch_target.to(DEVICE),
            )
            combined_scores = net(batch_imgs, batch_boxes, batch_scores)

            loss = WSDDN.calculate_loss(combined_scores, batch_target[0])
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()

        if epoch % STATE_PERIOD == 0:
            path = os.path.join(BASE_DIR, "states", f"epoch_{epoch}.pt")
            torch.save(
                net.state_dict(), path
            )
            tqdm.write(f"State saved to {path}")

        tqdm.write(f"Avg loss is {epoch_loss / len(train_ds)}")

        if epoch % EVAL_PERIOD == 0:
            tqdm.write(f"Evaluation started at {datetime.now()}")
            evaluate(net, test_dl)

        scheduler.step()
