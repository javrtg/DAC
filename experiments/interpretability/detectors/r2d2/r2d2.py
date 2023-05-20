from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tvf

from .tools.dataloader import norm_RGB
from .nets.patchnet import *


norm_RGB_mean = norm_RGB.transforms[1].mean
norm_RGB_std = norm_RGB.transforms[1].std
denorm_RGB = tvf.Normalize(
    mean = [-mi/stdi for mi, stdi in zip(norm_RGB_mean, norm_RGB_std)],
    std = [1./stdi for stdi in norm_RGB_std]
    )


def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    # print("\n>> Creating net = " + checkpoint['net']) 
    net = eval(checkpoint['net'])
    # nb_of_weights = common.model_size(net)
    # print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


class R2D2(torch.nn.Module):
    """ Reduced version of Superpoint that only returns the resized heatmap """
    def __init__(self, return_scores):
        super().__init__()
        self.net = load_network(
            str(Path(__file__).parent / 'models/r2d2_WASF_N16.pt'))
        
        mean, std = norm_RGB_mean, norm_RGB_std
        self.upper_bound = [(1 - mi) / stdi for mi, stdi in zip(mean, std)]
        self.lower_bound = [-mi / stdi for mi, stdi in zip(mean, std)]
        
        self.return_scores = return_scores
    
    def forward(self, x):
        res = self.net(imgs=[x])
        if self.return_scores:
            return res['repeatability'][0] * res['reliability'][0]
        else:
            return res['repeatability'][0]
        
    @staticmethod
    def preprocess(img):
        # gray -> rgb format
        if len(img.shape) == 2: 
            img = img[:,:,None]
        if img.shape[-1] == 1: 
            img = np.repeat(img, 3, axis=-1)
        return norm_RGB(img)
        
    @staticmethod
    def postprocess(tensor):
        with torch.no_grad():
            return (denorm_RGB(tensor[0]).cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
