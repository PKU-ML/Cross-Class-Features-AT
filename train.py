import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import argparse
from time import time
from utils import *
from model import PreActResNet18

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', required=True, type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', default='PRN', choices=['PRN'])
    parser.add_argument('--norm', default='linf', choices=['linf', 'l2'])
    parser.add_argument('--mode', choices=['AT', 'TRADES', 'none'], default='AT', type=str)
    parser.add_argument('--eps', default=-1., type=float)
    parser.add_argument('--alpha', default=-1., type=float)
    parser.add_argument('--steps', default=10, type=int)
    parser.add_argument('--beta', default=6, type=int)
    parser.add_argument('--test-eps', default=-1, type=float)
    parser.add_argument('--ne', default=200, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save-interval', default=10, type=int)
    parser.add_argument('--start', default=90, type=int)
    parser.add_argument('--end', default=110, type=int)
    parser.add_argument('--lam', default=0.8, type=float)
    parser.add_argument('--T', default=1.5, type=float)
    parser.add_argument('--decay', default=0.001, type=float)
    parser.add_argument('--no-rand-init', action='store_true')
    return parser.parse_args()

def lr_schedule(epoch, args):
    if args.ne == 100:
        if epoch < args.ne * 0.75:
            return args.lr
        elif epoch < args.ne * 0.9:
            return args.lr * 0.1
        else:
            return args.lr * 0.01
    elif args.ne == 200:
        if epoch < args.ne * 0.5:
            return args.lr
        elif epoch < args.ne * 0.75:
            return args.lr * 0.1
        else:
            return args.lr * 0.01
    else:
        return args.lr

def lam_schedule(epoch, args):
    if epoch > args.end:
        return args.lam
    else:
        return (epoch-args.start) / (args.end-args.start) * args.lam

def decay_schedule(epoch, args):
    return 1 - args.decay

if __name__ == '__main__':
    args = get_args()
    fname = 'train_log/' + args.fname
    os.makedirs(fname, exist_ok=True)
    norm = args.norm
    if args.eps < 0:
        args.eps = 128. if norm == 'l2' else 8.
    if args.alpha < 0:
        args.alpha = 16. if norm == 'l2' else 2.
    if args.test_eps < 0:
        args.test_eps = 128. if norm == 'l2' else 8.
    with open(f'{fname}/config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    dataset = args.dataset
    device = f'cuda:{args.device}'
    if args.model == 'PRN':
        model = PreActResNet18(10 if args.dataset == 'cifar10' else 100).to(device)
    else:
        raise NotImplementedError
    params = model.parameters()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_loader, test_loader = load_dataset(dataset, args.bs)
    eps = args.eps / 255.
    alpha = args.alpha / 255.
    test_eps = args.test_eps / 255.
    steps = args.steps
    test_alpha = test_eps / 4 if norm == 'linf' else test_eps / 8
    best_acc = 0
    normalize = normalize_cifar if args.dataset == 'cifar10' else normalize_cifar100
    all_log = []
    mode = args.mode
    if args.mode == 'AT':
        pgd = PGD(steps, alpha, eps, norm, False, normalize)
        if args.no_rand_init:
            pgd.rand_init = False
    elif args.mode == 'TRADES':
        pass
    elif args.mode == 'none':
        pass
    else:
        raise NotImplementedError
    for epoch in range(0, args.ne):
        start_time = time()
        log_data = np.zeros(6)
        lr = lr_schedule(epoch, args)
        opt.param_groups[0].update(lr=lr)
        train_bar = tqdm(train_loader)
        for x, y in train_bar:
            model.train()
            x, y = x.to(device), y.to(device)
            if mode == 'AT':
                delta = pgd.perturb(model, x, y)
                output = model(normalize(x + delta))
                if epoch >= args.start:
                    lam = lam_schedule(epoch, args)
                    loss = criterion(output, y)
                else:
                    loss = criterion(output, y)
            elif mode == 'none':
                output = model(normalize(x))
                loss = criterion(output, y)
            elif mode == 'TRADES':
                loss, output, x_adv = trades_loss(model, x, y, eps, alpha, steps, norm, opt, args.beta, True)
            if mode != 'none' and mode != 'TRADES':
                opt.zero_grad()
                loss.backward()
                opt.step()
            log_data[1] += (output.max(1)[1] == y).float().sum().item()
            log_data[2] += len(x)
            clean_output = model(normalize(x)) if mode != 'none' else output
            log_data[0] += (clean_output.max(1)[1] == y).float().sum().item()
            train_bar.set_description(
                f'Epoch {epoch}: Train Clean {log_data[0]/log_data[2]*100:.2f} Robust {log_data[1]/log_data[2]*100:.2f}'
            )
            decay = decay_schedule(epoch, args)
            model.eval()
            if args.debug:
                break
        model.eval()
        test_bar = tqdm(test_loader)
        test_pgd = PGD(20 if args.mode != 'none' else 1, test_alpha, test_eps, norm, False, normalize)
        for x, y in test_bar:
            x, y = x.to(device), y.to(device)
            clean_output = model(normalize(x)).detach()
            delta = test_pgd.perturb(model, x, y)
            output = model(normalize(x+delta))
            log_data[3] += (clean_output.max(1)[1] == y).float().sum().item()
            log_data[4] += (output.max(1)[1] == y).float().sum().item()
            log_data[5] += len(x)
            test_bar.set_description(
                f'Epoch {epoch}: Test Clean {log_data[3]/log_data[5]*100:.2f} Robust {log_data[4]/log_data[5]*100:.2f}'
            )
            if args.debug:
                break
        if log_data[4] > best_acc:
            torch.save(model.state_dict(), f'{fname}/model_best.pth')
            best_acc = log_data[4]
        if args.save_interval > 0 and (epoch+1)%args.save_interval == 0:
            torch.save(model.state_dict(), f'{fname}/model_{epoch+1}.pth')
            torch.save(opt.state_dict(), f'{fname}/opt_{epoch+1}.pth')
        all_log.append(log_data)
        all_log_df = np.stack(all_log, axis=0)
        df = pd.DataFrame(all_log_df)
        df.to_csv(f'{fname}/log.csv')
        torch.save(model.state_dict(), f'{fname}/model_last.pth')
