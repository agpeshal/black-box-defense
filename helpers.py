import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import foolbox.attacks as fa
from foolbox.models import PyTorchModel
from foolbox.criteria import Misclassification

class Boundary():
    def __init__(self, net, device='cuda'):
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.noise = self.random((128, 3, 32, 32))
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)

    def normalize(self, inputs):

        return (inputs - self.mean) / self.std

    def denormalise(self, inputs):
        return inputs * self.std + self.mean

    def gradient(self, inputs, targets):
        inputs.requires_grad_()
        outputs = self.net.eval()(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        grad = inputs.grad
        # print(grad)
        # zero_gradients(inputs)
        inputs.requires_grad = False
        self.net.zero_grad()

        grad = grad / torch.norm(grad.view(grad.size(0), -1), dim=1)
        # print(grad.shape)
        return grad

    def import_net(self, path):
        checkpoint = torch.load(path)
        self.net = nn.DataParallel(self.net)
        self.net.load_state_dict(checkpoint['net'])
        self.net.eval()
        self.net = self.net.to(self.device)

    def random(self, shape):
        dist = torch.distributions.normal.Normal(0, 1)
        z = dist.rsample(shape).to(self.device)
        z = z / z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]
        return z

    def cosine(self, inputs, targets, h=1):
        batch_size = inputs.size(0)
        normal_boundary = self.gradient(inputs, targets)
        z = self.random(inputs.shape)
        inputs_pert = inputs + h * z / self.std
        out = self.net(inputs_pert)
        print("True label", targets)
        print("New label", out.argmax(dim=1))
        normal_pert = self.gradient(inputs_pert, targets)

        dot_prod = torch.bmm(normal_boundary.view(batch_size, 1, -1),
                             normal_pert.view(batch_size, -1, 1))
        norm_boundary = torch.norm(normal_boundary.view(batch_size, -1), dim=1)
        norm_pert = torch.norm(normal_pert.view(batch_size, -1), dim=1)

        return dot_prod.view(-1) / (norm_boundary * norm_pert)

    def deepfool(self, inputs, targets):
        batch_size = inputs.size(0)
        inputs = self.denormalise(inputs)
        inputs = torch.clamp(inputs, 0, 1)
        preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
        fmodel = PyTorchModel(self.net, bounds=(0, 1), preprocessing=preprocessing, device=self.device)
        attack = fa.L2DeepFoolAttack(steps=50)
        _, clipped, _ = attack(fmodel, inputs, targets, epsilons=None)

        perturbation = (clipped - inputs)
        norm = torch.norm(perturbation.view(batch_size, -1), dim=1)
        direction = perturbation / norm  # Normalizing direction vector

        return direction, clipped

    def normal_deepfool(self, inputs, targets):

        output = self.net.eval()(inputs)
        _, prediction = output.max(1)

        # print("Predicted label", prediction)

        # Notice that here we do not provide the true labels to deepfool
        # Instead the current prediction since we are interested in the boundary!
        _, perturbed_image = self.deepfool(inputs, prediction)

        perturbed_image.requires_grad_()
        output = self.net.eval()(self.normalize(perturbed_image))
        _, perturbed_class = output.max(1)

        # print("Adversarial Class", perturbed_class)
        # print(output)
        diff = output[0][prediction.item()] - output[0][perturbed_class[0].item()]

        normal = torch.autograd.grad(diff, perturbed_image, create_graph=True)[0]
        perturbed_image.detach_()
        self.net.zero_grad()
        return normal / normal.norm(), perturbed_image

    def normal_hsja(self, inputs, iterations=20):

        output = self.net.eval()(inputs)
        _, prediction = output.max(1)

        # print("Predicted label", prediction)

        # Notice that here we do not provide the true labels to hsja
        # Instead the current prediction since we are interested in the boundary!
        perturbed_direc, perturbed_image = self.hsja(inputs, prediction, iterations=iterations)

        perturbed_image.requires_grad_()
        output = self.net.eval()(self.normalize(perturbed_image))
        _, perturbed_class = output.max(1)

        # print("Adversarial Class", perturbed_class)
        # print(output)
        diff = output[0][perturbed_class[0].item()] - output[0][prediction.item()]

        normal = torch.autograd.grad(diff, perturbed_image, create_graph=True)[0]
        perturbed_image.detach_()
        self.net.zero_grad()
        return normal / normal.norm(), perturbed_image, perturbed_direc

    def grad_estimate(self, inputs, targets, initial_evals=50, hz = 1.):
        # SOMETHING WRONG HERE>
        # Assuming input of batch size = 1 !!!!!
        c = inputs.shape[1]
        h = inputs.shape[2]
        w = inputs.shape[3]
        rv = self.random((initial_evals, c, h, w))

        delta = 200 * hz / (c * h  * w)

        output = self.net(inputs + delta * rv / self.std)
        _, prediction = output.max(1)
        prediction_bin = (prediction != targets).type(torch.float)
        base = prediction_bin.mean()
        grad_sum = 0
        for i in range(initial_evals):
            grad_sum += (prediction_bin[i] - base) * rv[i]
        grad = grad_sum / (initial_evals - 1)
        return grad / grad.norm()



    def boundary(self, inputs, targets):
        batch_size = inputs.size(0)
        normal_boundary = self.gradient(inputs, targets)
        norm_boundary = torch.norm(normal_boundary.view(batch_size, -1), dim=1)
        normal_boundary /= norm_boundary                                # Normalize the direction
        return normal_boundary

    def normal(self, inputs):
        # Assuming batch of size 1 !!!!!!!!!!!!!!
        # Assuming that the input is already at the boundary!
        inputs.requires_grad_()
        output = self.net.eval()(inputs)[0]
        sort, indexes = torch.sort(output, descending=True)
        adv_class = indexes[1]

       
        output[adv_class].backward()
        grad = inputs.grad
        self.net.zero_grad()
        inputs.requires_grad = False

        return grad / torch.norm(grad)

    def corr(self, inputs, targets, h):

        d1, b1 = self.normal_deepfool(inputs, targets)
        d2, b2 = self.normal_deepfool(inputs + h * self.random(inputs.shape) /self.std, targets)
        
        return torch.sum(d1*d2)

    def hsja(self, inputs, targets, iterations=20):
        '''
        Args:
            inputs: Batch of input images (pre-processed)
            targets: batch of labels
            iterations: HSJA steps

        Returns: Adversarials and its normalised directions

        '''
        preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
        fmodel = PyTorchModel(self.net, bounds=(0, 1), preprocessing=preprocessing, device=self.device)
        attack = fa.hop_skip_jump.HopSkipJump(steps=iterations, max_gradient_eval_steps=1000)
        images = self.denormalise(inputs)

        _, advs, _ = attack(fmodel, images, targets, epsilons=None)

        direc = advs - images
        norm = direc.norm()

        return direc / norm, advs


    def curvature(self, inputs, targets, h = 3.):
        '''
        Curvature of the boundaryy
        '''
        repeat = 1
        reg_sum = torch.tensor(0., device=self.device)
        for _ in range(repeat):
            # z = self.random(inputs.shape)
            z = self.noise[:inputs.size(0)]
            z, _ = self._find_z(inputs, targets, h)
            inputs.requires_grad_()
            outputs_orig = self.net.eval()(inputs)
            loss_orig = self.criterion(outputs_orig, targets)

            outputs_pos = self.net.eval()(inputs + h * z)
            loss_pos = self.criterion(outputs_pos, targets)
            grad_diff = torch.autograd.grad((loss_pos - loss_orig), inputs,
                                            grad_outputs=None, create_graph=True)[0]


            reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1) / h
            self.net.zero_grad()
            inputs.requires_grad = False
            inputs.detach_()
            reg_sum += torch.sum(reg)

        return reg_sum / repeat


    def plot(self, inputs, directions, save_location=None):
        dist = torch.distributions.normal.Normal(0, 1)
        rand = dist.rsample((3, 32, 32)).to(self.device)
        rand = rand / rand.norm()
        print("Norm", rand.norm())
        print("Dot product", torch.sum(rand * directions[0]).item())
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        xx, yy = np.meshgrid(x, y)
        grid = np.c_[xx.ravel(), yy.ravel()]

        with torch.no_grad():
            for i, image in enumerate(inputs):
                z = []
                for x, y in grid:
                    pert = rand * x + directions[i] * y
                    pert = pert / self.std[0]     # normalize the perturbation
                    image_pert = (image + pert).to(self.device)
                    out = self.net(image_pert.unsqueeze(0))
                    z.append(out.argmax(dim=1).cpu().item())
                # print(set(z))
                z = np.reshape(z, xx.shape)
                plt.contourf(xx, yy, z, levels=range(11))
                plt.colorbar()
                plt.scatter(0, 0, c='white')
                if save_location:
                    plt.savefig(save_location)
                plt.show()
                plt.close("all")
                plt.clf()
                break
