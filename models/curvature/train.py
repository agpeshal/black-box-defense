#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, "../../") # for utils!!

from CURE import CURELearner
from utils.utils import read_vision_dataset
from resnet import ResNet18


trainloader, testloader = read_vision_dataset('../../data', batch_size=128)

network = ResNet18()

net_CURE = CURELearner(network, trainloader, testloader, lambda_=2, device='cuda')
net_CURE.set_optimizer(optim_alg='Adam', args={'lr':1e-4}, scheduler='StepLR',
                       args_scheduler={'step_size':10, 'gamma':0.4})
net_CURE.import_model('../base/ckpt.pth')
# net_CURE.import_model('../cure/robust.pth')

h = [0.5]

net_CURE.train(epochs=40, h=h)
