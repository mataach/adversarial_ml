import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, utils
from torch.utils.data import TensorDataset, DataLoader
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import time
import h5py
from pathlib import Path
import os
import matplotlib.pylab as pl

from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = True



(x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()

x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=6)

test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
test_dataloader = DataLoader(test_dataset, batch_size=1000)
test_dataloader_single =  DataLoader(test_dataset, batch_size=1, num_workers=6)




# Training
def train(net, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        target = np.argmax(target, axis=1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = net(data).to(device)
        loss = F.nll_loss(F.log_softmax(prediction, dim=0), target)
        loss.backward()
        optimizer.step()

    
# Testing
def test(net):
    correct = 0
    with torch.no_grad():
        net.eval()
        for data, target in test_dataloader:
            output = net(data.to(device))
            pred = output.data.max(1, keepdim=True)[1].to("cpu")
            target = np.argmax(target, axis=1)
            correct += pred.eq(target.data.view_as(pred)).sum()
        acc_test = float(correct.numpy() / len(test_dataloader.dataset))
     
    print('Test accuracy: ', 100.*acc_test)
    return acc_test


def adjust_learning_rate(model, optimizer, epoch):
    if epoch == 30:
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif epoch == 40:
        return optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    return optimizer

