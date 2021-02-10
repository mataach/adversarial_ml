
import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

from utils_acts import RN50_acts, eval_acts
from utils_attack import perlin
from utils_model import loader_imgnet


dir_data = '/data/ktc116/data/ilsvrc2012/val'
dir_acts = './output_acts/imagenet/'
dir_perturbs = '/data/ktc116/adversarial-patch/patches_mod/'


def gen_fixed(patch, x_val):
    all_random_x = {}
    all_random_y = {}
    
    patch_shape = patch.shape
    
    # get dummy image
    x = torch.zeros_like(x_val)
    image_size = x.shape[-1]
    
    # get shape
    m_size = patch_shape[-1]
    for i in range(x.shape[0]):
        
        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)
        
        all_random_x[i] = random_x
        all_random_y[i] = random_y
        x[i, :, random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch
        
    mask = x.clone()
    mask[mask != 0] = 1.0
    
    return mask, all_random_x, all_random_y


def gen_mask(patch, x_val, fixed_mask, all_random_x, all_random_y):
    # get dummy image
    patch_shape = patch.shape
    x = torch.zeros_like(fixed_mask)
    image_size = x.shape[-1]
    
    # get shape
    m_size = patch_shape[-1]
    for i in range(x.shape[0]):
        random_x = all_random_x[i]
        random_y = all_random_y[i]
        # apply patch to dummy image  
        x[i, :, random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch
        
    return x

def init_circle(image_size, patch_size):
    image_size = image_size**2
    noise_size = int(image_size*patch_size)
    radius = int(math.sqrt(noise_size/math.pi))
    
    a = np.zeros((radius*2, radius*2))
    cx, cy = radius, radius # The center of circle 
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x**2 + y**2 <= radius**2
    a[cy-radius:cy+radius, cx-radius:cx+radius][index] = 1
    idx = np.flatnonzero((a == 0).all((1)))
    a = np.delete(a, idx, axis=0)
    a = np.expand_dims(a, axis = 0).repeat(3, axis = 0)
    
    return a, torch.FloatTensor(a)