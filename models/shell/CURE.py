import torch
from torch.autograd.gradcheck import zero_gradients
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
from torch.distributions import uniform
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class CURELearner():
    def __init__(self, net, trainloader, testloader, device='cuda', lambda_ = 1e-5, r = 1,
                 path='.'):
        '''
        CURE Class: Implementation of "Robustness via curvature regularization, and vice versa"     
                    in https://arxiv.org/abs/1811.09716
        ================================================
        Arguments:

        net: PyTorch nn
            network structure
        trainloader: PyTorch Dataloader
        testloader: PyTorch Dataloader
        device: 'cpu' or 'cuda' if GPU available
            type of decide to move tensors
        lambda_: float
            power of regularization
        path: string
            path to save the best model
        '''
        if not torch.cuda.is_available() and device=='cuda':
            raise ValueError("cuda is not available")

        self.net = net.to(device)
        # if device == 'cuda':
        self.net = nn.DataParallel(self.net)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.lambda_ = lambda_
        self.trainloader, self.testloader = trainloader, testloader
        self.path = path
        self.radius = r
        self.noise = self._random_z((512, 3, 32, 32))
        self.std = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
        self.test_acc_best = 0
        self.train_loss, self.train_acc, self.train_curv = [], [], []
        self.test_loss, self.test_acc_adv, self.test_acc_clean, self.test_curv = [], [], [], []
        self.writer = SummaryWriter("./runs/" + str(datetime.now()) + "_" + str(self.lambda_))

    def set_optimizer(self, optim_alg='Adam', args={'lr':1e-4}, scheduler=None, args_scheduler={}):
        '''
        Setting the optimizer of the network
        ================================================
        Arguments:
        
        optim_alg : string
            Name of the optimizer
        args: dict
            Parameter of the optimizer
        scheduler: optim.lr_scheduler
            Learning rate scheduler
        args_scheduler : dict
            Parameters of the scheduler
        '''
        self.optimizer = getattr(optim, optim_alg)(self.net.parameters(), **args)
        if not scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        else:
            self.scheduler = getattr(optim.lr_scheduler, scheduler)(self.optimizer, **args_scheduler)
        
            
    def train(self, h = [3], epochs = 15):
        '''
        Training the network
        ================================================
        Arguemnets:

        h : list with length less than the number of epochs
            Different h for different epochs of training, 
            can have a single number or a list of floats for each epoch
        epochs : int
            Number of epochs
        '''
        if len(h)>epochs:
            raise ValueError('Length of h should be less than number of epochs')
        if len(h)==1:
            h_all = epochs * [h[0]]
        else:
            h_all = epochs * [1.0]
            h_all[:len(h)] = list(h[:])
            h_all[len(h):] = (epochs - len(h)) * [h[-1]]

        for epoch, h_tmp in enumerate(h_all):
            self._train(epoch+1, h=h_tmp)
            self.test(epoch+1, h=h_tmp)
            self.scheduler.step()

    def _train(self, epoch, h):
        '''
        Training the model 
        '''
        print('\nEpoch: %d' % epoch)
        train_loss, total = 0, 0
        num_correct = 0
        reg_loss1, reg_loss2, net_loss = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            total += targets.size(0)
            outputs = self.net.eval()(inputs)
            noise = self._random_z(inputs.shape)

            # Check if the noise correlates with the direction of the gradient
            # positively or not. If not reverse its direction
            grad = self.gradient(inputs, targets)
            batch = inputs.size(0)
            dot_prod = torch.bmm(grad.view(batch, 1, -1), noise.view(batch, -1, 1))
            align = torch.sign(dot_prod.view(-1)).reshape((batch,1, 1, 1))
            noise = noise * align

            output1 = self.net.eval()(inputs + h * noise / self.std)
            output2 = self.net.eval()(inputs + (h+0.5) * noise / self.std)
            reg1 = self.criterion(output1, targets)
            reg2 = self.criterion(output2, (targets+1)%10)

            neg_log_likelihood = self.criterion(outputs, targets)
            loss = neg_log_likelihood + self.lambda_ * (reg1 + reg2)

            train_loss += neg_log_likelihood.item()
            reg_loss1 += reg1.item()
            reg_loss2 += reg2.item()
            net_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            _, predicted = outputs.max(1)
            outcome = predicted.data == targets
            num_correct += outcome.sum().item()


        self.writer.add_scalar("Train/CEloss", train_loss/(batch_idx+1), epoch)
        self.writer.add_scalar("Train/accuracy", num_correct/total, epoch)
        self.writer.add_scalar("Train/regularizer1", reg_loss1/(batch_idx+1), epoch)
        self.writer.add_scalar("Train/regularizer2", reg_loss2/(batch_idx+1), epoch)
        self.writer.add_scalar("Train/net_loss", net_loss/(batch_idx+1), epoch)

                
    def test(self, epoch, h):
        '''
        Testing the model 
        '''
        test_loss, total, reg_loss1, reg_loss2, net_loss, clean_acc = 0, 0, 0, 0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            with torch.no_grad():
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net.eval()(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                clean_acc += predicted.eq(targets).sum().item()
                total += targets.size(0)

            noise = self._random_z(inputs.shape)
            output1 = self.net.eval()(inputs + h * noise / self.std)
            output2 = self.net.eval()(inputs + (h + 0.5) * noise / self.std)
            reg1 = self.criterion(output1, targets)
            reg2 = self.criterion(output2, (targets + 1) % 10)

            reg_loss1 += reg1.item()
            reg_loss2 += reg2.item()
            net_loss += loss.item()

        print(f'epoch = {epoch},  clean_acc = {clean_acc/total}, loss = {test_loss/(batch_idx+1)}', \
            f'regularizer = {reg_loss1/(batch_idx+1)}')

        # self.test_loss.append(test_loss/(batch_idx+1))
        # self.test_acc_adv.append(100.*adv_acc/total)
        # self.test_acc_clean.append(100.*clean_acc/total)

        # if self.test_acc_clean[-1] > self.test_acc_best:
        if True:                                            # Always saving the current model
        #     self.test_acc_best = self.test_acc_clean[-1]
            print(f'Saving the best model to {self.path}')
            self.save_model(self.path + "/robust.pth")


        self.writer.add_scalar("Test/CEloss", test_loss/(batch_idx+1), epoch)
        self.writer.add_scalar("Test/accuracy", clean_acc/total, epoch )
        self.writer.add_scalar("Test/regularizer1", reg_loss1/(batch_idx+1), epoch)
        self.writer.add_scalar("Test/regularizer2", reg_loss2/(batch_idx+1), epoch)
        self.writer.add_scalar("Test/net_loss", net_loss/(batch_idx+1), epoch)

    
    def _random_z(self, shape):
        '''
        Finding a random direction
        '''
        dist = torch.distributions.normal.Normal(0, 1)
        z = dist.rsample(shape)
        z = z / z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]

        return z.to(self.device)

    def gradient(self, inputs, targets):
        inputs.requires_grad_()
        outputs = self.net.eval()(inputs)
        loss = self.criterion(outputs, targets)
        grad = torch.autograd.grad(loss, inputs, create_graph=True)[0]
        inputs.requires_grad = False
        self.net.zero_grad()
        # print(grad.shape)
        return grad

    def cosine(self, inputs, targets, h=1):
        batch_size = inputs.size(0)
        grad = self.gradient(inputs, targets)
        z = self._random_z(inputs.shape)
        inputs_pert = inputs + h * z / self.std
        grad_pert = self.gradient(inputs_pert, targets)

        dot_prod = torch.bmm(grad.view(batch_size, 1, -1),
                             grad_pert.view(batch_size, -1, 1))
        norm = torch.norm(grad.view(batch_size, -1), dim=1) + 1e-7
        norm_pert = torch.norm(grad_pert.view(batch_size, -1), dim=1) + 1e-7

        cosine_abs = torch.abs(dot_prod.view(-1) / (norm * norm_pert))

        if torch.isnan(cosine_abs).any():
            print(norm, norm_pert, dot_prod)

        return torch.mean(cosine_abs)


    def _find_z(self, inputs, targets, h):
        '''
        Finding the direction in the regularizer
        '''
        inputs.requires_grad_()
        outputs = self.net.eval()(inputs)
        loss_z = self.criterion(outputs, targets)
        loss_z.backward()
        grad = inputs.grad.data + 0.0
        norm_grad = grad.norm().item()
        z = torch.sign(grad).detach() + 0.
        z = 1.*(h) * (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)
        zero_gradients(inputs)
        self.net.zero_grad()

        return z, norm_grad
    
        
    def regularizer(self, inputs, targets, h = 3.):
        '''
        Regularizer term in CURE
        '''
        # z = self._random_z(inputs.shape)
        z = self.noise[:inputs.size(0)]
        # z, _ = self._find_z(inputs, targets, h)
        self.net.zero_grad()
        inputs.requires_grad_()
        outputs_orig = self.net.eval()(inputs)
        outputs_pos = self.net.eval()(inputs + h * z)

        loss_pos = self.criterion(outputs_pos, targets)
        loss_orig = self.criterion(outputs_orig, targets)
        grad_diff = torch.autograd.grad((loss_pos - loss_orig), inputs,
                                        grad_outputs=None, create_graph=True)[0]


        reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1) / h
        self.net.zero_grad()
        inputs.requires_grad = False
        inputs.detach_()
        # inputs = inputs.detach()
        # return torch.sum(reg) / float(inputs.size(0))
        return torch.sum(reg)

    def get_perturbations(self, size):
        dist = torch.distributions.normal.Normal(0, 1)
        samples = dist.rsample((size, 3, 32, 32))
        samples.squeeze_(dim=-1)  # rsample adds an additional shape 1 in the end
        norms = torch.norm(samples.view(size, -1), dim=1)
        norms = norms.reshape((size, 1, 1, 1))
        samples = samples / norms
        # Transform the noise just as the input so that shell radius do not transform
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        samples = samples / std.view(1, 3, 1, 1)
        return samples
            
    def save_model(self, path):
        '''
        Saving the model
        ================================================
        Arguments:

        path: string
            path to save the model
        '''
        
        print('Saving...')

        state = {
            'net': self.net.state_dict(),
            # 'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, path)
        
    def import_model(self, path):
        '''
        Importing the pre-trained model
        '''
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
           
            
    def plot_results(self):
        """
        Plotting the results
        """
        plt.figure(figsize=(15,12))
        plt.suptitle('Results',fontsize = 18,y = 0.96)
        plt.subplot(3,3,1)
        plt.plot(self.train_acc, Linewidth=2, c = 'C0')
        plt.plot(self.test_acc_clean, Linewidth=2, c = 'C1')
        plt.plot(self.test_acc_adv, Linewidth=2, c = 'C2')
        plt.legend(['train_clean', 'test_clean', 'test_adv'], fontsize = 14)
        plt.title('Accuracy', fontsize = 14)
        plt.ylabel('Accuracy', fontsize = 14)
        plt.xlabel('epoch', fontsize = 14) 
        plt.grid()  
        plt.subplot(3,3,2)
        plt.plot(self.train_curv, Linewidth=2, c = 'C0')
        plt.plot(self.test_curv, Linewidth=2, c = 'C1')
        plt.legend(['train_curv', 'test_curv'], fontsize = 14)
        plt.title('Curvetaure', fontsize = 14)
        plt.ylabel('curv', fontsize = 14)
        plt.xlabel('epoch', fontsize = 14)
        plt.grid()   
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.subplot(3,3,3)
        plt.plot(self.train_loss, Linewidth=2, c = 'C0')
        plt.plot(self.test_loss, Linewidth=2, c = 'C1')
        plt.legend(['train', 'test'], fontsize = 14)
        plt.title('Loss', fontsize = 14)
        plt.ylabel('loss', fontsize = 14)
        plt.xlabel('epoch', fontsize = 14)
        plt.grid()   
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()
        
                                                                                                                                                                                                                                                                                                                                                                                                                                       