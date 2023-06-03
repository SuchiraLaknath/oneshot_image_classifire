import torch
#from torch import Module
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights = "IMAGENET1K_V2")
        
    def forward(self, x):
        return self.model(x)

class Feature_Extractor(nn.Module):
    def __init__(self) :
        super().__init__()

        self.model = Model()
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, image1, image2):
        features1 = self.model(image1)
        features2 = self.model(image2)
        #features1 = torch.nn.Flatten(features1)
        #features2 = torch.nn.Flatten(features2)
        similarity = F.cosine_similarity(features1,features2)
        return similarity