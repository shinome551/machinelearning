#!/usr/bin/env python
# coding: utf-8

import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset


## P:lesson teacher's predict
## Q:answer student's predict
## kl(q||p) = sum( Q * log(Q / P) )
## F.kl_div(P.log(), Q, reduction='sum')  
def softmax_KLDiv(answer, lesson, T=1.0):
    return T * T * F.kl_div((answer / T).log_softmax(1), (lesson / T).softmax(1), reduction='batchmean')


def distillation(teacher, student, optimizer, trainloader, T, device):
    teacher.eval()
    student.train()
    trainloss = 0
    for data in trainloader:
        inputs, _ = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            lesson = teacher(inputs)
        answer = student(inputs)
        loss = softmax_KLDiv(answer, lesson, T)
        #loss = F.mse_loss(answer, lesson, reduction='mean')
        loss.backward()
        optimizer.step()
        trainloss += loss.item() * inputs.size()[0]

    trainloss = trainloss / len(trainloader.dataset)
    return trainloss


def softmax_JSDiv(answer, lesson, T=1.0, lmd= 0.5):
    Q = (answer / T).log_softmax(1).exp()
    P = (lesson / T).log_softmax(1).exp()
    M = (lmd * Q + (1. - lmd) * P).log()
    return T * T * (lmd * F.kl_div(M, Q, reduction='batchmean') + (1. - lmd) * F.kl_div(M, P, reduction='batchmean'))


def train(model, optimizer, trainloader, device):
    model.train()
    trainloss = 0
    for data in trainloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        trainloss += loss.item() * inputs.size()[0]

    trainloss = trainloss / len(trainloader.dataset)
    return trainloss


def test(model, testloader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / len(testloader.dataset)
    return acc


def getWeights_loss(model, loader, device):
    model.eval()
    weights = torch.tensor([])
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction='none')
            weights = torch.cat([weights, loss.data.cpu()])
    return weights


def getWeights_entropy(model, loader, device):
    model.eval()
    weights = torch.tensor([])
    with torch.no_grad():
        for data in loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            log_prob = F.log_softmax(outputs, 1)
            weights = torch.cat([weights, - torch.sum(log_prob * log_prob.exp(), 1).data.cpu()])
    return weights


def spearman_rank_corr(X, Y):
    N = len(X)
    rho = 1 - 6 * ((X - Y) ** 2).sum().float() / (N ** 3 - N)
    return rho


def weight2index(weights, idx_pool, sample_num, train_tag):
    _, indices = torch.sort(weights, descending=True)
    select_idx = idx_pool[indices[:sample_num]]
    _, feq = torch.unique(train_tag[select_idx], return_counts=True)
    print(feq)
    return select_idx


def make_class_balanced_random_idx(sample_num, train_tag):
    class_num = len(torch.unique(train_tag))
    num_percls = sample_num // class_num
    rand_idx = torch.LongTensor([])
    for c in range(class_num):
        idx_percls = torch.nonzero(train_tag == c)
        rand_idx_percls = idx_percls[torch.randperm(len(idx_percls))][:num_percls]
        rand_idx = torch.cat((rand_idx, rand_idx_percls))
    return rand_idx[:, 0]


def save_var(filepath, **kwargs):
    torch.save(kwargs, filepath)

    
def image_grid(imgs, filename):
    imgs_np = np.array(imgs) / 255.0
    imgs_np = np.transpose(imgs_np, [0,3,1,2])
    imgs_th = torch.as_tensor(imgs_np)
    torchvision.utils.save_image(imgs_th, filename,
                                 nrow=10, padding=5)

    
## データセットのインデックスを取得できるようにデータセットを拡張するクラス
## 叩くとデータ・ラベル・インデックスを返す
class IndexLapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target, idx

    def __len__(self):
        return len(self.dataset)

    
def train_and_ret_pred(model, optimizer, trainloader, device):
    model.train()
    trainloss = 0
    predicts = torch.tensor([])
    indices_all = torch.LongTensor([]) 
    trainloader_with_idx = DataLoader(IndexLapper(trainloader.dataset), batch_size=trainloader.batch_size, shuffle=True)
    for data in trainloader_with_idx:
        inputs, labels, indices = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        trainloss += loss.item() * inputs.size()[0]
        ## return predicts and indices
        pred = outputs.data.log_softmax(1).exp()
        predicts = torch.cat([predicts, pred.data.cpu()], 0)
        indices_all = torch.cat([indices_all, indices], 0)
    
    trainloss = trainloss / len(trainloader.dataset)
    return trainloss, predicts, indices_all


def farthest_first_traversal(embedding, n_cluster):
    centroids_idx = torch.LongTensor([0])
    for _ in range(n_cluster-1):
        v, _ = torch.cdist(embedding, embedding[centroids_idx]).min(1)
        idx = v.argmax(0, keepdim=True).cpu()
        centroids_idx = torch.cat([centroids_idx, idx])
    return centroids_idx


def postprocess_by_fft(embedding, sample_idx, train_tag, rate):
    new_sample_idx = torch.LongTensor([])
    sample_idx_bool = torch.zeros_like(train_tag==0)
    sample_idx_bool[sample_idx] = True
    for t in train_tag.unique():
        indices = torch.nonzero((train_tag == t) * sample_idx_bool)[:,0]
        indices_clswis = indices[farthest_first_traversal(embedding[indices], int(rate*len(indices)))]
        new_sample_idx = torch.cat([new_sample_idx, indices_clswis])
    return new_sample_idx


def calcECE(model, loader, bin_size=0.1, T=1):
    model.eval()
    stats = {'pred':[],
             'true':[],
             'conf':[]}
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            stats['true'].extend(labels.unsqueeze(1).tolist())
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) / T
            conf, pred = outputs.log_softmax(1).exp().topk(1)
            stats['pred'].extend(pred.tolist())
            stats['conf'].extend(conf.tolist())

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)

    stats_pred = np.array(stats['pred'])
    stats_true = np.array(stats['true'])
    stats_conf = np.array(stats['conf'])

    binned_confs = np.digitize(stats_conf, upper_bounds, right=True)
    uni = np.unique(binned_confs)
    ECE = 0
    for i, upper_bound in enumerate(upper_bounds):
        acc = np.mean((stats_pred == stats_true)[binned_confs == i]) if i in uni else 0
        conf = np.mean(stats_conf[binned_confs == i]) if i in uni else 0
        n_b = np.sum([binned_confs == i])
        ECE += np.abs(acc - conf) * n_b

    return ECE / len(loader.dataset)


def smooth_one_hot(labels, num_classes, factor):
    assert 0 <= factor < 1
    labels_shape = torch.Size((labels.size(0), num_classes))
    with torch.no_grad():
        smooth_one_hot = torch.empty(size=labels_shape, device=labels.device)
        smooth_one_hot.fill_(factor / (num_classes - 1))
        smooth_one_hot.scatter_(1, labels.data.unsqueeze(1), 1.0 - factor)
    return smooth_one_hot

def train_ls(model, optimizer, trainloader, device, num_classes, alpha):
    model.train()
    trainloss = 0
    for data in trainloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        label_smooth = smooth_one_hot(labels, num_classes, alpha)
        loss = F.kl_div(outputs.log_softmax(1), label_smooth, reduction='batchmean')
        loss.backward()
        optimizer.step()
        trainloss += loss.item() * inputs.size()[0]

    trainloss = trainloss / len(trainloader.dataset)
    return trainloss


def distillation_ls(label_smoothing, student, optimizer, trainloader, T, device):
    student.train()
    trainloss = 0
    for data in trainloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        answer = student(inputs)
        lesson = label_smoothing(labels)
        loss = T * T * F.kl_div((answer / T).log_softmax(1), lesson, reduction='batchmean')
        loss.backward()
        optimizer.step()
        trainloss += loss.item() * inputs.size()[0]

    trainloss = trainloss / len(trainloader.dataset)
    return trainloss
