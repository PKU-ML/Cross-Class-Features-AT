import torch
from model import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
from torchvision import datasets, transforms

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model2', default='None', type=str)
    parser.add_argument('--arch', choices=['PRN18', 'RN18'], default='PRN18')
    parser.add_argument('--norm', choices=['l2', 'linf'], default='linf')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--eps', default=8., type=float)
    parser.add_argument('--vmin', default=0., type=float)
    parser.add_argument('--vmax', default=1., type=float)
    return parser.parse_args()

def get_feature(loader, model, adv=True, adv_config=None, activate=False, normalize=False, num_classes=10):
    X = [x for x,_ in loader]
    Y = [y for _,y in loader]
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    normalization = normalize_cifar if normalize is False else normalize
    all_features = []
    acc, cnt = 0, 0
    if adv:
        pgd = PGD(20, adv_config['alpha'], adv_config['eps'], adv_config['norm'], False, normalization)
    for i in tqdm(range(num_classes)):
        yi = Y == i
        xi = X[yi]
        features = []
        bs = len(xi) // 100
        for k in range(bs):
            x = xi[100*k: 100*(k+1)].clone()
            x = x.cuda()
            y = Y[yi][100*k: 100*(k+1)]
            delta = pgd.perturb(model, x, y.cuda())
            f = model.get_feature(normalization(x+delta), i if not activate else None).detach()
            features.append(f)
            y = y.cuda()
            acc += (model(normalization(x+delta)).max(1)[1] == y).float().sum()
            cnt += len(y)
        features = torch.cat(features, dim=0).cpu()
        all_features.append(features)
    return all_features, acc/cnt

def cw_corelation(features, vmin=-1, vmax=1, index=None, save_name=None, activate=None):
    cwf = []
    for i in range(len(features)):
        data = features[i]
        data = data.mean(0)
        data = data / torch.norm(data,p=2)
        cwf.append(data)
    cwc = torch.zeros(len(features),len(features))
    for i in range(len(features)):
        for j in range(len(features)):
            cwc[i, j] = (cwf[i] * cwf[j]).sum()
    if activate is not None:
        cwc = activate(cwc)
    if index is not None:
        cwc = cwc[index]
        cwc = cwc[:, index]
    plt.imshow(cwc, cmap='Blues', vmin=vmin, vmax=vmax)
    plt.colorbar()
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, dpi=200,bbox_inches='tight')
    plt.clf()
    return cwc

def show_diff(cb, cl, save_name=None, scale=0.5):
    diff = cb-cl
    plt.imshow(diff, cmap='Blues', vmin=0, vmax=scale)
    plt.colorbar()
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, dpi=200,bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    args = get_args()
    num_classes = 10 if args.dataset == 'cifar10' else 100
    model_name = args.model
    model = PreActResNet18(num_classes) if (args.arch == 'PRN18') else ResNet18(num_classes)
    ckpt = torch.load(model_name, map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    model.to('cuda:0')
    if args.model2 != 'None':
        model2_name = args.model2
        model2 = PreActResNet18(num_classes) if (args.arch == 'PRN18') else ResNet18(num_classes)
        ckpt = torch.load(model2_name, map_location='cpu')
        model2.load_state_dict(ckpt)
        model2.eval()
        model2.to('cuda:0')
    os.makedirs('figs/'+args.fname, exist_ok=True)
    os.makedirs('features/'+args.fname, exist_ok=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/data/cifar_data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False
    ) if args.dataset == 'cifar10' else torch.utils.data.DataLoader(
        datasets.CIFAR100('/data/cifar_data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False
    )
    adv_config = {
        'eps': args.eps / 255. if args.norm == 'linf' else 128./255.,
        'alpha': (args.eps /255.) / 4. if args.norm == 'linf' else 16./255.,
        'norm': args.norm
    }
    test_adv, test_acc = get_feature(test_loader, model, True, adv_config, num_classes=num_classes)
    torch.save(test_adv, f'features/{args.fname}/test_adv.pth')
    if args.model2 != 'None':
        cwc = cw_corelation(test_adv, args.vmin, args.vmax, None, None)
        test_adv, test_acc = get_feature(test_loader, model2, True, adv_config, num_classes=num_classes)
        cwc2 = cw_corelation(test_adv, args.vmin, args.vmax, None, None)
        show_diff(cwc, cwc2, 'figs/'+args.fname+f'/{args.fname}.png')
    else:
        cwc = cw_corelation(test_adv, args.vmin, args.vmax, None, 'figs/'+args.fname+f'/{args.fname}.png')
        sum_cwc = torch.clamp(cwc, 0, 1).sum().item() - len(cwc)
        print(f'{args.fname}: test acc = {test_acc*100:.2f}%\t sum={sum_cwc}')
