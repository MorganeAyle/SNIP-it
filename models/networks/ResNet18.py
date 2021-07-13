from models.Pruneable import Pruneable
from models.networks.assisting_layers.ResNetLayers import resnet18


class ResNet18(Pruneable):
    def __init__(self, device="cuda", output_dim=2, input_dim=(1, 1, 1), **kwargs):
        super(ResNet18, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)
        self.m = resnet18()

    def forward(self, x):
        x = self.m(x)
        return x
