'''
Similar in structure and style to utils_model for ImageNet
Functions for:
- Loading models
- Making data loader
- Eval on clean
- Eval on uap
'''

import multiprocessing
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torchvision

sys.path.append('.')
sys.path.append('./model_softprune')

from torch.utils import model_zoo
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

import model_cifar
import model_prune
from model_quant.resnet_quant import resnet20
from model_prune.utils_prune import *
from model_softprune.model import Net


CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_STD_Q = [0.247, 0.243, 0.262]

SVHN_MEAN = [0.4376821, 0.4437697, 0.47280442]
SVHN_STD = [0.19803012, 0.20101562, 0.19703614]

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


'''
# Load pre-trained ResNet18 models
# Non-quantized models

dataset
- cifar10
- svhn

model_name
- resnet18
- [soft-pruning]
    - resnet18_sfP
    - resnet18_sfP-mixup
    - resnet18_sfP-cutout
- [post-training pruning]
    resnet18_PX_0.Y
    X = 2, 3, 4
    Y = 0, 3, 6, 9
- [quantization]
    resnet20_QX
    X = 2, 3, 4
'''
def get_model(model_name, dataset, batch_size = None, ckpt_dir = 'checkpoint/'):
    
    ckpt_file = '%s/%s.pth' % (dataset, model_name)
    print('Loading ' + ckpt_file)
    
    # Regular model
    if model_name == 'resnet18':
        model = model_cifar.ResNet18()
        assert os.path.isdir(ckpt_dir), 'Error: no checkpoint directory found!'
        if dataset == 'cifar10': checkpoint = torch.load(ckpt_dir + ckpt_file)['net']
        else: checkpoint = torch.load(ckpt_dir + ckpt_file)['state_dict']
        
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint)
    
    # Quantization
    elif model_name[:10] == 'resnet20_Q':
        Q_param = int(model_name[-1])
        model = resnet20()
        model = torch.nn.DataParallel(model)
        model_params = []
        for name, params in model.module.named_parameters():
            if 'act_alpha' in name:
                model_params += [{'params': [params], 'lr': 1e-1, 'weight_decay': 1e-4}]
            elif 'wgt_alpha' in name:
                model_params += [{'params': [params], 'lr': 2e-2, 'weight_decay': 1e-4}]
            else:
                model_params += [{'params': [params]}]
        model.load_state_dict(torch.load(ckpt_dir + ckpt_file)['state_dict'], strict = False)
        
    # Post-training pruning
    elif model_name[:10] == 'resnet18_P':
        layer = int(model_name[10]) # Should be 2, 3, or 4; e.g. 'resnet18_P3'
        prune_pct = float(model_name[-3:])
        
        if dataset == 'cifar10':
            if batch_size == None: batch_size = 500
            model = model_prune.ResNet18(batch_size = batch_size)
            checkpoint = torch.load(ckpt_dir + dataset + '/resnet18.pth')['net']
        else:
            if batch_size == None: batch_size = 1627
            model = model_prune.ResNet18(batch_size = batch_size)
            checkpoint = torch.load(ckpt_dir + dataset + '/resnet18.pth')['state_dict']
        
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint)
        
        # Load and apply masks
        masks = load_masks(layer, dataset = dataset)
        net_mask = layer_to_mask(model, layer)
        kernel_shape = net_mask.shape[1:]
        
        # Assign mask to corresponding layer
        mask_reshaped = masks[prune_pct].view(kernel_shape)
        if layer == 2:
            model.module.mask_2 = mask_reshaped
        elif layer == 3:
            model.module.mask_3 = mask_reshaped
        elif layer == 4:
            model.module.mask_4 = mask_reshaped
    
    # Soft filter pruning
    elif model_name[:12] == 'resnet18_sfP':   
        args = torch.load('./args/' + ckpt_file)
        model = Net(arch = args.arch, criterion = nn.CrossEntropyLoss(), args = args)
        checkpoint = torch.load(ckpt_dir + ckpt_file)
        
        model.load_state_dict(checkpoint)
        model = torch.nn.DataParallel(model)
    
    # Normalization wrapper
    # Hack so that I can just use dataloder for non-normalized data
    if dataset == 'cifar10' and model_name[:10] == 'resnet20_Q':
        data_mean, data_std = [CIFAR_MEAN, CIFAR_STD_Q]
    elif dataset == 'cifar10':
        data_mean, data_std = [CIFAR_MEAN, CIFAR_STD]
    elif dataset == 'svhn':
        data_mean, data_std = [SVHN_MEAN, SVHN_STD]
        
    normalize = Normalizer(mean = data_mean, std = data_std)
    model = nn.Sequential(normalize, model)
    model = model.cuda()
    
    return model


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)
    
    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    
    
def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize
    """
    # We assume the color channel is at dim = 1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


def get_testdata(dataset, batch_size = None, normalize = False):

    if dataset == 'cifar10':
        dir_data = './data/cifar10'
        if batch_size == None: batch_size = 500
        testloader = loader_cifar(dir_data, train = False, batch_size = batch_size, normalize = normalize)
        return testloader
    elif dataset == 'svhn':
        dir_data = './data/svhn'
        if batch_size == None: batch_size = 1627
        testloader = loader_svhn(dir_data, split = 'test', batch_size = batch_size, normalize = normalize)
        return testloader

    
# DataLoader for CIFAR10 dataset
def loader_cifar(dir_data, train = False, batch_size = 500, normalize = False):
    transform_test = transform(normalize, CIFAR_MEAN, CIFAR_STD)
    dataset = torchvision.datasets.CIFAR10(root = dir_data, train = train, download = True, transform = transform_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = max(1, multiprocessing.cpu_count() - 1))
    return dataloader


# DataLoader for SVHN dataset
def loader_svhn(dir_data, split = 'test', batch_size = 500, normalize = False):
    transform_test = transform(normalize, SVHN_MEAN, SVHN_STD)
    dataset = torchvision.datasets.SVHN(root=dir_data, split=split, download=True, transform=transform_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=max(1, multiprocessing.cpu_count() - 1))
    return dataloader


def transform(normalize, MEAN, STD):
    if normalize:
         transform_test = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(mean = MEAN, std = STD),
         ])
    else:
        transform_test = transforms.Compose([transforms.ToTensor()])
    return transform_test


def evaluate(model, loader, uap = None, verbose = True):
    probs, labels = [], []
    model.eval()
    
    clamp_min, clamp_max = [0, 1]
    
    if uap is not None:
        _, (x_val, y_val) = next(enumerate(loader))
        batch_size = len(x_val)
        uap = uap.unsqueeze(0).repeat([batch_size, 1, 1, 1])
    
    with torch.set_grad_enabled(False):
        for i, (x_val, y_val) in enumerate(loader):
            
            if uap is None:
                out = torch.nn.functional.softmax(model(x_val.float().cuda()), dim = 1)
            else:
                perturbed = torch.clamp((x_val + uap).cuda(), clamp_min, clamp_max)
                out = torch.nn.functional.softmax(model(perturbed), dim = 1)
                
            probs.append(out.cpu().numpy())
            labels.append(y_val)
            
            if verbose and (i % 20 == 0): print("\r Completed: {:.2%}".format(i / len(loader), end = ''), flush = True)
            sys.stdout.flush()
            
    # Convert batches to single numpy arrays    
    probs = np.stack([p for l in probs for p in l])
    labels = np.array([t for l in labels for t in l])
    
    # Extract top 5 predictions for each example
    n = 5
    top = np.argpartition(-probs, n, axis = 1)[:,:n]
    top_probs = probs[np.arange(probs.shape[0])[:, None], top].astype(np.float16)
    top1acc = top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels
    top5acc = [labels[i] in row for i, row in enumerate(top)]
    outputs = top[range(len(top)), np.argmax(top_probs, axis = 1)]
        
    return top, top_probs, top1acc, top5acc, outputs, labels
