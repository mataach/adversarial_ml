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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = True



# Training
def train(net, optimizer, train_dataloader):
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
def test(net, test_dataloader):
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