import argparse

import torch
from torch.utils.data import DataLoader

from datasets import VOCandSSW
from network import WSDDN
from utils import evaluate, evaluate_detectron2

SCALES = [480, 576, 688, 864, 1200]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--path", help="Path of trained model's state")
    args = parser.parse_args()

    net = WSDDN()
    net.load_state_dict(torch.load(args.path))
    net.to(DEVICE)

    print("State is loaded")

    test_ds = VOCandSSW("test", SCALES)  # len = 4952
    test_dl = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=1)

    # evaluate_detectron2(net, test_dl)
    evaluate(net, test_dl)
