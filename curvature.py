
import sys
from CURE.CURE import CURELearner
import matplotlib.pyplot as plt
from utils.utils import read_vision_dataset
from resnet import ResNet18
import torch
import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='base', type=str,
                    help='model name to be loaded')
parser.add_argument('--batch_size', type=int, default=20,
                    help='Batch size')
parser.add_argument('--hz', type=float, default=0.5,
                    help='Magnitude of noise for curvature calc')
parser.add_argument('--seed', type=int, default=4,
                    help='Random Seed')

sys.path.insert(0, "../")

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainloader, testloader = read_vision_dataset('./data', batch_size=args.batch_size)
network = ResNet18()

net_CURE = CURELearner(network, trainloader, testloader, device=device)
model = args.model


if args.model == 'base':
    net_CURE.import_model('./models/base/ckpt.pth')
else:
    net_CURE.import_model('./models/'+model+'/robust.pth')

fname = "Curvature_batch_" + str(args.batch_size) + ".txt"

f = open('./models/'+ model + '/' + fname, "w")
f.write("Batch Size: {}\n".format(args.batch_size))

h = args.hz

curv = 0
for batch_idx, (images, labels) in enumerate(testloader):
    images = images.to(net_CURE.device)
    labels = labels.to(net_CURE.device)
    curv += net_CURE.regularizer(images, labels, h).item()
    break

print("Average Curvature on {} Test images {:.4f}".format(args.batch_size, curv / (batch_idx+1)))
f.write("Average Curvature on Test images {:.4f}\n".format(curv / (batch_idx+1)))

curv = 0
for batch_idx, (images, labels) in enumerate(trainloader):
    images = images.to(net_CURE.device)
    labels = labels.to(net_CURE.device)
    curv += net_CURE.regularizer(images, labels, h).item()
    break

print("Average Curvature on {} Train images {:.4f}".format(args.batch_size, curv / (batch_idx+1)))
f.write("Average Curvature on Train images {:.4f}\n".format(curv / (batch_idx+1)))
f.close()
