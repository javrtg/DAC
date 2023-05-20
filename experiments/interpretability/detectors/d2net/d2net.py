from pathlib import Path

import numpy as np
# import cv2
import torch

from .lib.model_test import D2Net as D2N
from .lib.utils import preprocess_image


class D2Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = D2N(
            model_file=str(Path(__file__).parent / "models/d2_tf.pth"),
            use_relu=True,
            use_cuda=True
        )

        self.upper_bound = [255 - 103.939, 255 - 116.779, 255 - 123.68]
        self.lower_bound = [-103.939, -116.779, -123.68]

    def forward(self, x):
        # We hardcode here the channel since after exploring d2net predictions
        # we noted that channel 360 usually produces the highest number of
        # detections. Therefore we use this channel to amplify the confidence
        # measurements.
        channel = 360
        return self.model.dense_feature_extraction(x)[:, channel:channel + 1]

    @staticmethod
    def preprocess(img):
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.repeat(img, 3, -1)

        # we skip resizing since we are using small images
        resized_image = img
        # if max(resized_image.shape) > max_edge:
        #     factor = max_edge / max(resized_image.shape)
        #     resized_image = cv2.resize(resized_image,
        #                     dsize=None, fx=factor, fy=factor).astype('float')
        # if sum(resized_image.shape[: 2]) > max_sum_edges:
        #     factor = max_sum_edges / sum(resized_image.shape[: 2])
        #     resized_image = cv2.resize(resized_image,
        #                     dsize=None, fx=factor, fy=factor).astype('float')

        return torch.from_numpy(
            preprocess_image(resized_image, preprocessing='caffe')).float()  # (C, H, W)

    @staticmethod
    def postprocess(tensor):
        # inverse caffe preprocessing
        mean = np.array([103.939, 116.779, 123.68])
        return (
            (tensor[0].detach().cpu().numpy()
             + mean.reshape([3, 1, 1]))[::-1].transpose(1, 2, 0)
        ).astype(np.uint8)
