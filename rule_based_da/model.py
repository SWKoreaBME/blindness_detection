from torch import nn
import torchvision.models as models

import torch

def classifier():
    return models.inception_v3(pretrained=True)

# def classifier():
#     return models.resnet18(pretrained=True)

if __name__ == "__main__":
    classifier()