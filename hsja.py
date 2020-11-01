import torch
from torchvision import transforms
from utils.utils import read_vision_dataset
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
parser.add_argument('--model', default='different_label', type=str,
                    help='model name to be loaded')
parser.add_argument('--iter', default=10, type=int,
                    help='Number of HSJA iterations')
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

trainloader, testloader = read_vision_dataset('./data', batch_size=args.batch_size,
                                              transform=transform)
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
attack = fa.hop_skip_jump.HopSkipJump(steps=args.iter, max_gradient_eval_steps=1000, tensorboard=None)

adv_distances = []

location = './models/' + model
fname = "HSJA_new_iter=" + str(args.iter) + ".png"

for images, labels in testloader:

    images = images.to(device)
    labels = labels.to(device)
    print("True labels", labels)

    _, advs, _ = attack(fmodel, images, labels, epsilons=None)
    dist = torch.norm((advs-images).view(len(images), -1), dim=1)
    print(dist)
    dist = dist[dist > 1e-4].cpu().numpy()

    adv_distances.extend(dist)

    break

print("Mean distance", np.mean(adv_distances))

plt.hist(adv_distances)
plt.title('Median distance {:.3f}'.format(np.median(adv_distances)))
plt.savefig(os.path.join(location, fname))
plt.show()
