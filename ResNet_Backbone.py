import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet18(weights='DEFAULT')
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Extract up to the last conv layer

    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == "__main__":
    print("ResNet Backbone model class is ready.")
