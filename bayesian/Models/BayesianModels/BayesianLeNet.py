import math
import torch.nn as nn
from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import BBB_MGP_Linear, BBB_MGP_Conv2d
from layers import FlattenLayer, ModuleWrapper


class BBBLeNet(ModuleWrapper):
    '''The architecture of LeNet with bayesian Layers'''

    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBLeNet, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        elif layer_type=='mgp':
            BBBLinear = BBB_MGP_Linear
            BBBConv2d = BBB_MGP_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(inputs, 6, 5, padding=0, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(6, 16, 5, padding=0, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BBBConv2d(16, 120, 5, padding=0, bias=True, priors=self.priors)
        self.act3 = self.act()
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))

        self.flatten = FlattenLayer(3 * 3 * 120)
        self.fc1 = BBBLinear(3 * 3 * 120, 84, bias=True, priors=self.priors)
        self.act4 = self.act()

        # self.fc2 = BBBLinear(120, 84, bias=True, priors=self.priors)
        # self.act4 = self.act()

        self.fc2 = BBBLinear(84, outputs, bias=True, priors=self.priors)
