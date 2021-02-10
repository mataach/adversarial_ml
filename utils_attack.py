import math
import numpy as np
import torch
import torch.nn as nn
from noise import pnoise2

'''
UAP Attacks
1) SGD-UAP
    1.1 untargetd
    1.2 targeted
    1.3 layer

Fix!
2) Perlin Noise

Pending...?
3) Random Noise
4) Gabor Noise ---?

5) CD-UAP ---?
'''


# Adapted from https://github.com/BorealisAI/advertorch/blob/master/advertorch/attacks/iterative_projected_gradient.py
# If we want targeted, y_val must be the target labels, y_target = -1 to indicate no target

# layer_name - UAP Layer Attack
# y_target - Targeted UAP Attack

def uap_batch(model, loader, nb_epoch, eps, beta = 8, y_target = -1, layer_name = None, cut_end = False):
        
    _, (x_val, y_val) = next(enumerate(loader))
    batch_size = len(x_val)
    batch_delta = torch.randn_like(x_val).sign() * eps / 2 # initialize as a vector with values {-eps, eps}
    delta = batch_delta[0]
    losses = []
    
    # Loss function
    if layer_name is None:
        loss_fn = nn.CrossEntropyLoss(reduction = 'none')
        beta = torch.cuda.FloatTensor([beta])
        def clamped_loss(output, target):
            loss = torch.mean(torch.min(loss_fn(output, target), beta))
            return loss
    else:
        def get_norm(self, forward_input, forward_output):
            global main_value
            main_value = torch.norm(forward_output, p = 'fro')
        for name, layer in model.named_modules():
            if name == layer_name:
                handle = layer.register_forward_hook(get_norm)
                
    batch_delta.requires_grad_()
    for epoch in range(nb_epoch):
        eps_step = eps / 5
        
        for i, (x_val, y_val) in enumerate(loader):
            if cut_end and (len(x_val) < batch_size): break
                
            if y_target >= 0: y_val = torch.ones(size = y_val.shape, dtype = y_val.dtype) * y_target
            perturbed = torch.clamp((x_val + batch_delta).cuda(), 0, 1)
            outputs = model(perturbed)
            if i % 20 == 0: print("\r Completed: {:.2%}".format((epoch * len(loader) + i) / (nb_epoch * len(loader)), end = ''), flush = True)
            
            # Loss function value
            if layer_name is None: loss = clamped_loss(outputs, y_val.cuda())
            else: loss = main_value
            
            if y_target >= 0: loss = -loss
            losses.append(torch.mean(loss))
            loss.backward()
            
            # Batch update
            grad_sign = batch_delta.grad.data.mean(dim = 0).sign()
            delta = delta + grad_sign * eps_step 
            delta = torch.clamp(delta, -eps, eps)
                
            batch_delta.data = delta.unsqueeze(0).repeat([batch_size, 1, 1, 1])
            batch_delta.grad.data.zero_()
    
    if layer_name is not None: handle.remove() # release hook
    
    return delta.data, losses


# Generate perlin noise UAP
'''
Using preprocessing steps from [pn4]
frac are all normalized from 0 to 1
frac_x, frac_y, frac_sin, octave = noise_params[xx]
freq_x = 1 / (frac_x * 224 / 3 + 224 / 10)
freq_y = 1 / (frac_y * 224 / 3 + 224 / 10)
freq_sine = frac_sin * 224 / 5 + 224 / 100
params = [1 / freq_x, 1 / freq_y, freq_sine, octave]
'''
def perlin(size_x, size_y, params):
    period_x, period_y, freq_sine, octave = params
    noise = torch.zeros(torch.Size([size_x, size_y]))
    for x in range(size_x):
        for y in range(size_y):
            noise[x][y] = pnoise2(x / period_x, y / period_y, octaves = int(octave))
    noise = (noise - noise.min()) / (noise.max() - noise.min())    
    noise = (torch.sin(noise * freq_sine * math.pi)).sign()
    noise = noise.unsqueeze(0).repeat([3, 1, 1])
    return noise

# Random attack

def random():
    pass

# CD-UAP attack
# 
def uap_cd():
    pass
