import torch
import torch.nn as nn
import torch.optim as optim
from model import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
from torchvision import datasets, transforms

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--arch', choices=['PRN18', 'RN18'], default='PRN18')
    parser.add_argument('--norm', choices=['l2', 'linf'], default='linf')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--eps', default=8., type=float)
    return parser.parse_args()

def get_feature(loader, model, adv=False, adv_config=None, activate=False, normalize=False):
    X = [x for x,_ in loader]
    Y = [y for _,y in loader]
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    normalization = normalize_cifar if normalize is False else normalize
    all_features = []
    if adv:
        pgd = PGD(20, adv_config['alpha'], adv_config['eps'], adv_config['norm'], False, normalization)
    for i in tqdm(range(10 if normalization == normalize_cifar else 100)):
        yi = Y == i
        xi = X[yi]
        features = []
        bs = len(xi) // 100
        for k in range(bs):
            x = xi[100*k: 100*(k+1)].clone()
            x = x.cuda()
            if not adv:
                f = model.get_feature(normalization(x), i if not activate else None).detach()
            else:
                y = Y[yi][100*k: 100*(k+1)]
                delta = pgd.perturb(model, x, y.cuda())
                f = model.get_feature(normalization(x+delta), i if not activate else None).detach()
            features.append(f)
        features = torch.cat(features, dim=0).cpu()
        all_features.append(features)
    return all_features

if __name__ == '__main__':
    args = get_args()
    model_name = args.model
    model = PreActResNet18(10 if args.dataset == 'cifar10' else 100) if (args.arch == 'PRN18') else ResNet18(10 if args.dataset == 'cifar10' else 100)
    ckpt = torch.load(model_name, map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    model.to('cuda:0')
    os.makedirs('figs/'+args.fname, exist_ok=True)
    os.makedirs('features/'+args.fname, exist_ok=True)
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/data/cifar_data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False
    ) if args.dataset == 'cifar10' else torch.utils.data.DataLoader(
        datasets.CIFAR100('/data/cifar_data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/data/cifar_data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False
    ) if args.dataset == 'cifar10' else torch.utils.data.DataLoader(
        datasets.CIFAR100('/data/cifar_data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False
    )
    test_f = get_feature(test_loader, model)
    torch.save(test_f, f'features/{args.fname}/test_clean.pth')
    adv_config = {
        'eps': args.eps / 255. if args.norm == 'linf' else 128./255.,
        'alpha': (args.eps /255.) / 4. if args.norm == 'linf' else args.eps / 8,
        'norm': args.norm
    }
    test_adv = get_feature(test_loader, model, True, adv_config)
    torch.save(test_adv, f'features/{args.fname}/test_adv.pth')
