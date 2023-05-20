import numpy as np
from pathlib import Path

import cv2
import torch


class SuperPoint(torch.nn.Module):
    """ Reduced version of Superpoint that only returns the resized heatmap """
    def __init__(self, return_scores):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        
        # load weights
        self.load_state_dict(torch.load(
            str(Path(__file__).parent / 'superpoint_v1.pth')
            ))
        
        # image value limits accordinf to the preprocessing (will be used to
        # clamp the updates of the tensor)
        self.upper_bound = 1.
        self.lower_bound = 0.
        
        self.return_scores = return_scores
        
    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Shared Encoder.
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Detector Head.
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
       
        # softmax in cell dimension (channels)
        if self.return_scores:
            scores = torch.nn.functional.softmax(scores, dim=1)
        # remove dustbin:
        scores = scores[:,:-1]
        
        # reshape the cells to image space dimensions:
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        return scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)[None]
    
    @staticmethod
    def preprocess(img, size=(640,480), closest=True):
        """
        Superpoint specific preprocessing (we define it as staticmethod to 
        facilitate importing)
        
        Parameters
        ----------
        img : ndarray (H,W,C)
            input image.
        size : tuple of two ints, optional
            Image size prior to feed it to SP. The default is (640,480) which 
            is used in the paper and recommended in:
            https://github.com/magicleap/SuperPointPretrainedNetwork/issues/5#issuecomment-529297976
        closest : bool, optional
            If True, the image will be resized to the closest spatial 
            dimensions that are divisible by 8, ignoring size argument. 
            Default: True.

        Returns
        -------
        out : float32 ndarray (H,W,C).
            Resized image.
        """
        # convert to grayscale if needed:
        if len(img.shape)==3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
        interp = cv2.INTER_AREA
        if not closest:
            # check valid resize
            assert (size[1]%8)==0, f"h: {size[1]} is not divisible by 8."
            assert (size[0]%8)==0, f"w: {size[0]} is not divisible by 8."
            if img.shape[:2]==size[::-1]:
                out = img
            else:
                out = cv2.resize(img, size, interpolation=interp)
        else:
            # make image sizes divisible by 8 (SP's requirement)
            h, w = img.shape[:2]
            hmod, wmod = h%8, w%8
            h_new = (h-hmod if hmod<4 else h+hmod)
            w_new = (w-wmod if wmod<4 else w+wmod)
            if (hmod | wmod):
                out = cv2.resize(img, (w_new, h_new), interpolation=interp)
            else:
                out = img
        out = out.astype(np.float32) / 255.
        return out 
    
    @staticmethod
    def postprocess(tensor):
        """Processed tensor -> Numpy and convert values to a [0,255] scale"""
        return (255 * tensor[0,0].detach().to('cpu').numpy()).astype(np.uint8)
