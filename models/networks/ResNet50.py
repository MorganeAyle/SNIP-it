from models.Pruneable import Pruneable
from models.networks.assisting_layers.ResNetLayers import resnet50


class ResNet50(Pruneable):
    def __init__(self, device="cuda", output_dim=100, input_dim=(1, 1, 1), **kwargs):
        super(ResNet50, self).__init__(device=device, output_dim=output_dim, input_dim=input_dim, **kwargs)
        print(output_dim)
        self.m = resnet50(output_dim)

    def forward(self, x):
        x = self.m(x)
        return x

