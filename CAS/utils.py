import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
 

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
mu = torch.tensor(CIFAR10_MEAN).view(3,1,1)
std = torch.tensor(CIFAR10_STD).view(3,1,1)
def normalize_cifar(x):
    return (x - mu.to(x.device))/(std.to(x.device))

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
mu_cifar100 = torch.tensor(CIFAR100_MEAN).view(3,1,1)
std_cifar100 = torch.tensor(CIFAR100_STD).view(3,1,1)
def normalize_cifar100(x):
    return (x - mu_cifar100.to(x.device))/(std_cifar100.to(x.device))

def load_dataset(dataset='cifar10', batch_size=128, test_transform=False):
    if dataset == 'cifar10':
        transform_ = transforms.Compose([transforms.ToTensor()])
        if test_transform:
            train_transform_ = transforms.Compose([transforms.ToTensor()])
        else:
            train_transform_ = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/data/cifar_data', train=True, download=True, transform=train_transform_),
            batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/data/cifar_data', train=False, download=True, transform=transform_),
            batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader
    elif dataset == 'cifar100':
        transform_ = transforms.Compose([transforms.ToTensor()])
        if test_transform:
            train_transform_ = transforms.Compose([transforms.ToTensor()])
        else:
            train_transform_ = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('/data/cifar_data', train=True, download=True, transform=train_transform_),
            batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('/data/cifar_data', train=False, download=True, transform=transform_),
            batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader
    else:
        raise NotImplementedError

def trades_loss(model, x, y, eps, alpha, n_iters, norm, optimizer, beta, return_adv=False):
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x)
    x_adv = x.detach() + 0.001 * torch.randn(x.shape).to(x.device).detach()
    if norm == 'linf':
        for _ in range(n_iters):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(normalize_cifar(x_adv)), dim=1),
                                       F.softmax(model(normalize_cifar(x)), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif norm == 'l2':
        delta = 0.001 * torch.randn(x.shape).to(x.device).detach()
        delta = Variable(delta.data, requires_grad=True)
        optimizer_delta = optim.SGD([delta], lr=eps / n_iters * 2)
        for _ in range(n_iters):
            adv = x + delta
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(normalize_cifar(adv)), dim=1),
                                           F.softmax(model(normalize_cifar(x)), dim=1))
            loss.backward()
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()
            delta.data.add_(x)
            delta.data.clamp_(0, 1).sub_(x)
            delta.data.renorm_(p=2, dim=0, maxnorm=eps)
        x_adv = Variable(x + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    logits = model(normalize_cifar(x))
    loss_natural = F.cross_entropy(logits, y)
    adv_output = model(normalize_cifar(x_adv))
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_output, dim=1),
                                                    F.softmax(logits, dim=1))
    loss = loss_natural + beta * loss_robust
    if return_adv:
        return loss, adv_output, x_adv
    return loss, adv_output

class Attack():
    def __init__(self, iters, alpha, eps, norm, criterion, rand_init, rand_perturb, targeted, normalize=normalize_cifar):
        self.iters = iters
        self.alpha = alpha
        self.eps = eps
        self.norm = norm
        self.criterion = criterion
        self.rand_init = rand_init
        self.rand_perturb = rand_perturb
        self.targetd = targeted
        self.normalize = normalize
    def perturb(self, model, x, y):
        delta = torch.zeros_like(x).to(x.device)
        if self.rand_init:
            if self.norm == "linf":
                delta.uniform_(-self.eps, self.eps)
            elif self.norm == "l2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*self.eps
            else:
                raise ValueError
        delta = torch.clamp(delta, 0-x, 1-x)
        delta.requires_grad = True
        for _ in range(self.iters):
            output = model(self.normalize(x+delta))
            loss = self.criterion(output, y)
            if self.targetd:
                loss *= -1
            loss.backward()
            g = delta.grad.detach()
            if self.norm == "linf":
                d = torch.clamp(delta + self.alpha * torch.sign(g), min=-self.eps, max=self.eps).detach()
            elif self.norm == "l2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (delta + scaled_g*self.alpha).view(delta.size(0),-1).renorm(p=2,dim=0,maxnorm=self.eps).view_as(delta).detach()
            d = torch.clamp(d, 0 - x, 1 - x)
            delta.data = d
            delta.grad.zero_()
        return delta.detach()

class PGD(Attack):
    def __init__(self, iters, alpha, eps, norm, targeted=False, normalize=normalize_cifar):
        super().__init__(iters, alpha, eps, norm, nn.CrossEntropyLoss(), True, False, targeted, normalize=normalize)
