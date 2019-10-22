import argparse

import torch
from torch.utils.data import DataLoader

from datasets import VOCandSSW
from network import WSDDN
from utils import evaluate

SCALES = [480, 576, 688, 864, 1200]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("path", help="Path of trained model's state")
    args = parser.parse_args()

    net = torch.load(args.path)
    print("State is loaded")

    test_ds = VOCandSSW("test", SCALES)  # len = 4952
    test_dl = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=1)

    evaluate(net, SCALES, test_dl)
