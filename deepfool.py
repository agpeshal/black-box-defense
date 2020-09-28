import torch
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn

from resnet import ResNet18
import argparse
import numpy as np

import foolbox.attacks as fa
from foolbox.models import PyTorchModel

import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=20,
                    help='Training data batch size')
parser.add_argument('--model', default='base', type=str,
                    help='model name to be loaded')
parser.add_argument('-seed', type=int, default=4,
                    help='Seed for consistency')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

transform = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

net = ResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

model = args.model
if model == 'base':
    checkpoint = torch.load('./models/base/ckpt.pth')
else:
    checkpoint = torch.load('./models/'+model+'/robust.pth')

net.load_state_dict(checkpoint['net'])
net.eval()

preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
fmodel = PyTorchModel(net, bounds=(0, 1), preprocessing=preprocessing, device=device)
attack = fa.L2DeepFoolAttack(steps=10)

adv_distances = []

location = './models/' + model
fname = "DeepFool.png"

for images, labels in testloader:

    images = images.to(device)
    labels = labels.to(device)

    _, advs, _ = attack(fmodel, images, labels, epsilons=None)
    dist = torch.norm((advs-images).view(len(images), -1), dim=1)

    dist = dist[dist > 1e-4].cpu().numpy()

    adv_distances.extend(dist)

    break

print("Mean distance", np.mean(adv_distances))

plt.hist(adv_distances)
plt.title('Median distance {:.3f}'.format(np.median(adv_distances)))
plt.savefig(os.path.join(location, fname))
plt.show()
