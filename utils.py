#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset


# In[12]:


## P:lesson teacher's predict
## Q:answer student's predict
## kl(q||p) = sum( Q * log(Q / P) )
## F.kl_div(P.log(), Q, reduction='sum')  
def softmax_KLDiv(answer, lesson, T=1.0):
    return T * T * F.kl_div((answer / T).log_softmax(1),                             (lesson / T).softmax(1), reduction='batchmean')

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


# In[13]:


def softmax_JSDiv(answer, lesson, T=1.0, lmd= 0.5):
    Q = (answer / T).log_softmax(1).exp()
    P = (lesson / T).log_softmax(1).exp()
    M = (lmd * Q + (1. - lmd) * P).log()
    return T * T * (lmd * F.kl_div(M, Q, reduction='batchmean') +                     (1. - lmd) * F.kl_div(M, P, reduction='batchmean'))


# In[14]:


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


# In[15]:


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


# In[1]:


def weight2index(weights, idx_pool, sample_num, train_tag):
    _, indices = torch.sort(weights, descending=True)
    select_idx = idx_pool[indices[:sample_num]]
    _, feq = torch.unique(train_tag[select_idx], return_counts=True)
    print(feq)
    return select_idx


# In[21]:


def make_class_balanced_random_idx(sample_num, train_tag):
    class_num = len(torch.unique(train_tag))
    num_percls = sample_num // class_num
    rand_idx = torch.LongTensor([])
    for c in range(10):
        idx_percls = torch.nonzero(train_tag == c)
        rand_idx_percls = idx_percls[torch.randperm(len(idx_percls))][:num_percls]
        rand_idx = torch.cat((rand_idx, rand_idx_percls))
    return rand_idx[:, 0]


# In[20]:


def save_dict(name, filepath, state_dict, trainloss, accuracy, sample_idx, T):
    d = {}
    d['mode'] = name
    d['state_dict'] = state_dict
    d['trainloss'] = trainloss
    d['accuracy'] = accuracy
    d['sample_idx'] = sample_idx.tolist()
    d['T'] = T
    torch.save(d, filepath + '_'.join([name, str(len(sample_idx)), str(T)]))


# In[19]:


def image_grid(imgs, filename):
    imgs_np = np.array(imgs) / 255.0
    imgs_np = np.transpose(imgs_np, [0,3,1,2])
    imgs_th = torch.as_tensor(imgs_np)
    torchvision.utils.save_image(imgs_th, filename,
                                 nrow=10, padding=5)


# In[ ]:




