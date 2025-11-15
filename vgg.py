import torch
import torch.nn as nn
from torchvision import models, transforms
import utils


class VGG16(nn.Module):
    def __init__(self, vgg_path=None):
        super(VGG16, self).__init__()

        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg16 = models.vgg16(weights=weights)

        # If user provided a custom vgg_path (caffe-converted), try loading it (non-strict)
        if vgg_path:
            try:
                vgg16.load_state_dict(torch.load(vgg_path), strict=False)
            except Exception as e:
                print("Warning: couldn't load custom vgg weights: ", e)

        self.features = vgg16.features

        # freeze gradients
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        layers = {"3": "relu1_2", "8": "relu2_2", "15": "relu3_3", "22": "relu4_3"}
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
                if name == "22":
                    break
        return features
