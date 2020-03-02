import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import VocAndEb
from network import WSDDN
from utils import evaluate

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--base_net", type=str, default="vgg", help="Base network to use"
    )
    parser.add_argument("--state_path", help="Path of trained model's state")
    args = parser.parse_args()

    net = WSDDN(base_net=args.base_net)
    net.load_state_dict(torch.load(args.state_path))
    net.to(DEVICE)

    tqdm.write("State is loaded")

    test_ds = VocAndEb("test")  # len = 4952
    test_dl = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=4)
    evaluate(net, test_dl)
