#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.insert(0, "models/curvature")  # for CURE!!

from CURE import CURELearner
from utils.utils import read_vision_dataset
from resnet import ResNet18
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, help="batch_size")
    parser.add_argument("--epochs", default=40, help="trainin gepochs")
    parser.add_argument("--lambda_", default=2, help="power of regularization")
    parser.add_argument("--lr", default=1e-4, help="learning rate")
    parser.add_argument("--step", default=10, help="scheduler step")
    parser.add_argument("--gamma", default=0.4, help="factor for lr scheduler")

    args = parser.parse_args()
    trainloader, testloader = read_vision_dataset("./data", batch_size=128)
    network = ResNet18()

    net_CURE = CURELearner(
        network, trainloader, testloader, lambda_=args.lambda_, device="cuda"
    )
    net_CURE.set_optimizer(
        optim_alg="Adam",
        args={"lr": args.lr},
        scheduler="StepLR",
        args_scheduler={"step_size": args.step, "gamma": args.gamma},
    )
    net_CURE.import_model("./base/ckpt.pth")
    # net_CURE.import_model('../cure/robust.pth')

    h = [0.5]

    net_CURE.train(epochs=args.epochs, h=h)


if __name__ == "__main__":
    main()
