import torch
#from torch import Module
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) :
        super().__init__()

        self.model = models.resnet50(weights = "IMAGENET1K_V2")
        self.model = nn.Sequential()

    def forward(self, image1, image2):
        features1 = self.model(image1)
        features2 = self.model(image2)

        similarity = F.cosine_embedding_loss(features1, features2)