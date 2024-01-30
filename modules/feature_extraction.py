# A simplified version of the original code - https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
import torch.nn as nn
from modules.cnn.unet import UNet

class UNet_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=1, output_channel=512):
        super(UNet_FeatureExtractor, self).__init__()
        self.ConvNet = UNet(input_channel, output_channel)

    def forward(self, input):
        return self.ConvNet(input)