# models/model.py
import torch
import torch.nn as nn
from .encoder import ImageEncoder
from .decoder import CaptionDecoder

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = ImageEncoder(embed_size)
        self.decoder = CaptionDecoder(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def generate_caption(self, image, vocab, max_length=20):
        features = self.encoder(image)
        return self.decoder.generate_caption(features, vocab, max_length)