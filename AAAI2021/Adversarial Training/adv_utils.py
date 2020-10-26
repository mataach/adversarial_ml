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

from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

from adv_attacks import EAD_L1, PGD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = True



(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=100, num_workers=6)

test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
test_dataloader = DataLoader(test_dataset, batch_size=1000, num_workers=6)



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



# Adversarial Training
def adv_train(net, optimizer, norm, eps, a):
    net.train()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        target = np.argmax(target, axis=1)   
        adv_batch = adversarial_batch(net, optimizer, data, norm, eps, a)
        batch, target = adv_batch.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = net(batch).to(device)
        loss = F.nll_loss(F.log_softmax(prediction, dim=0), target)
        loss.backward()
        optimizer.step()


def adversarial_batch(model, optimizer, data, norm, eps=0, a=0):
    data, batch = data.to(device), torch.Tensor().to(device)
    classifier = make_classifier(model, optimizer, i_shape=(1, 28, 28))
    if norm == 1:
        perturbed_image = EAD_L1(classifier, data.cpu().detach().numpy())
    elif norm == 2 or norm == np.inf:
        perturbed_image = PGD(classifier, data.cpu().detach().numpy(), norm, eps, a)
    else:
        raise Exception("Norms accepted are 1, 2 or np.inf")
    batch = torch.cat((batch, torch.Tensor(perturbed_image).to(device)), dim=0)
    
    return batch


def make_classifier(model, optimizer, i_shape):
    # Make a classifier wrapper!
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_, max_),
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=i_shape,
        nb_classes=10,
    )
    
    return classifier


def adjust_learning_rate(model, optimizer, epoch, dataset):
    if dataset == 'mnist' and epoch == 5:
        return optim.Adam(model.parameters(), lr=1e-4)
    elif dataset == 'cifar10':
        if epoch == 30:
            return optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        elif epoch == 40:
            return optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    
    return optimizer

