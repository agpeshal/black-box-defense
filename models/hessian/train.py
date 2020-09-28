#!/usr/bin/env python
# coding: utf-8

# # Robustness via curvature regularization, and vice versa
# This notebooks demonstrates how to use the CURE algorithm for training a robust network.

import sys
sys.path.insert(0, "../../") # for utils!!

from CURE import CURELearner
from utils.utils import read_vision_dataset
from resnet import ResNet18
import torch
import numpy as np
seed = 1

torch.manual_seed(seed)
np.random.seed(seed)

trainloader, testloader = read_vision_dataset('../../data', batch_size=128)

network = ResNet18()

net_CURE = CURELearner(network, trainloader, testloader, lambda_=0.005, eigen=10,
                       device='cuda')
net_CURE.set_optimizer(optim_alg='Adam', args={'lr':5e-5})
net_CURE.import_model('../base/ckpt.pth')
# net_CURE.import_model('../cure/robust.pth')

h = [0.01]

net_CURE.train(epochs=10, h=h)
