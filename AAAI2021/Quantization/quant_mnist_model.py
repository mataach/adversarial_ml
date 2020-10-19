import torch
import torch.nn as nn
import math
from QuantModules import *



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = first_conv(1, 10, kernel_size=5, stride=1, padding=0, bias=False)
        self.conv2 = QuantConv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = last_fc(320, 512)
        self.fc2 = last_fc(512, 200)
        self.fc3 = last_fc(200, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x    
    

def classifier():
    model = Classifier()
    return model