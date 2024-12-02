# models/encoder.py
import torch
import torch.nn as nn
import torchvision

class ImageEncoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(ImageEncoder, self).__init__()
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Add a new layer to get the embedding size we want
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        # Freeze the CNN parameters if not training
        if not train_CNN:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features