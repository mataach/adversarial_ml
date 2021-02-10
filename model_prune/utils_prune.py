import json
import torch

from .prune_resnet18 import *


use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


def load_masks(layer, prune_dir='model_prune/', dataset='cifar10', model='resnet18'):
    file = prune_dir + dataset + '/' + model + '_layer{}.json'.format(layer)
    with open(file, 'r') as fp:
        data = json.load(fp)
    masks = {float(pg): torch.tensor(mask).to(device) for pg,mask in data.items()}
    
    return masks


def layer_to_mask(net, layer):
    if layer == 2:
        return net.module.mask_2
    elif layer == 3:
        return net.module.mask_3
    elif layer == 4:
        return net.module.mask_4
    else:
        raise Exception("Layers accepted are 2, 3 or 4")


