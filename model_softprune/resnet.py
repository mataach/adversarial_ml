'''
This program implements the ResNet architecture.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202',
           'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockSmall(nn.Module):
    ''' A basic block for small ResNet architectures. '''
    expansion = 1

    def __init__(self, in_planes, planes, stride = 1, option = 'A'):
        super(BasicBlockSmall, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                # For CIFAR10 ResNet paper, uses option A.
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetSmall(nn.Module):
    ''' Small ResNet architectures. '''
    
    def __init__(self, block, num_blocks, num_classes = 10):
        super(ResNetSmall, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride = 2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def conv3x3(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride, padding = 1, bias = False)


class BasicBlockLarge(nn.Module):
    ''' A basic block for large ResNet architectures. '''
    expansion = 1

    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlockLarge, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlockLarge(nn.Module):
    ''' Pre-activation version of the BasicBlockLarge. '''
    expansion = 1

    def __init__(self, in_planes, planes, stride = 1):
        super(PreActBlockLarge, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size = 1, stride = stride, bias = False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class BottleneckLarge(nn.Module):
    ''' Bottleneck for large ResNet architectures. '''
    expansion = 4

    def __init__(self, in_planes, planes, stride = 1):
        super(BottleneckLarge, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneckLarge(nn.Module):
    ''' Pre-activation version of the original BottleneckLarge module. '''
    expansion = 4

    def __init__(self, in_planes, planes, stride = 1):
        super(PreActBottleneckLarge, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size = 1, bias = False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNetLarge(nn.Module):
    ''' Large ResNet architectures. '''
    
    def __init__(self, block, num_blocks, num_classes = 10):
        super(ResNetLarge, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin = 0, lout = 5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out



def ResNet(arch, num_classes = 10):
    ''' Constructs a ResNet model. '''
    if arch == 'resnet20':
        return ResNetSmall(BasicBlockSmall, [3, 3, 3], num_classes)
    elif arch == 'resnet32':
        return ResNetSmall(BasicBlockSmall, [5, 5, 5], num_classes)
    elif arch == 'resnet44':
        return ResNetSmall(BasicBlockSmall, [7, 7, 7], num_classes)
    elif arch == 'resnet56':
        return ResNetSmall(BasicBlockSmall, [9, 9, 9], num_classes)
    elif arch == 'resnet110':
        return ResNetSmall(BasicBlockSmall, [18, 18, 18], num_classes)
    elif arch == 'resnet1202':
        return ResNetSmall(BasicBlockSmall, [200, 200, 200], num_classes)

    elif arch == 'resnet18':
        return ResNetLarge(BasicBlockLarge, [2,2,2,2], num_classes)
    elif arch == 'resnet34':
        return ResNetLarge(BasicBlockLarge, [3,4,6,3], num_classes)
    elif arch == 'resnet50':
        return ResNetLarge(BottleneckLarge, [3,4,6,3], num_classes)
    elif arch == 'resnet101':
        return ResNetLarge(BottleneckLarge, [3,4,23,3], num_classes)
    elif arch == 'resnet152':
        return ResNetLarge(BottleneckLarge, [3,8,36,3], num_classes)

    else:
        raise Exception("Invalid architecture.")


def test():
    ''' Tests the implementation of ResNet. '''
    for arch in __all__: 
        net = ResNet(arch)
        y = net(Variable(torch.randn(1,3,32,32)))
        print(y.size())
        print()
        
#test()
