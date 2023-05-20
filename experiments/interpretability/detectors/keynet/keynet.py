from pathlib import Path

import cv2
import numpy as np
import torch

from .model.network import KeyNet as Keynet


class KeyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        keynet_model = Keynet(
            num_filters=8, 
            num_levels=3, 
            kernel_size=5
            )
        
        checkpoint = torch.load(
            str(Path(__file__).parent / 'model/weights/keynet_pytorch.pth'))
        keynet_model.load_state_dict(checkpoint['state_dict'])
        self.keynet_model = keynet_model
        
        self.upper_bound = 1.
        self.lower_bound = 0.
        
    def forward(self, x):
        return self.keynet_model(x)
    
    @staticmethod
    def preprocess(img):
        if len(img.shape)==3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return (img / 255).astype(np.float32)
    
    @staticmethod
    def postprocess(tensor):
        return (255 * tensor[0,0].detach().to('cpu').numpy()).astype(np.uint8)
