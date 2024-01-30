# A simplified version of the original code - https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
import torch.nn as nn
from modules.dropout_layer import dropout_layer
from modules.sequence_modeling import BidirectionalLSTM
from modules.feature_extraction import UNet_FeatureExtractor

class Model(nn.Module):

    def __init__(self, num_class=181, device='cpu'):
        super(Model, self).__init__()
        self.device = device

        """ FeatureExtraction """
        self.FeatureExtraction = UNet_FeatureExtractor(1, 512)
        self.FeatureExtraction_output = 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        
        """
        Temporal Dropout
        """
        self.dropout1 = dropout_layer(self.device)
        self.dropout2 = dropout_layer(self.device)
        self.dropout3 = dropout_layer(self.device)
        self.dropout4 = dropout_layer(self.device)
        self.dropout5 = dropout_layer(self.device)

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
            BidirectionalLSTM(256, 256, 256))
        self.SequenceModeling_output = 256

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input, text=None, is_train=True):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Temporal Dropout + Sequence modeling stage """
        visual_feature_after_dropout1 = self.dropout1(visual_feature)
        visual_feature_after_dropout2 = self.dropout2(visual_feature)
        visual_feature_after_dropout3 = self.dropout3(visual_feature)
        visual_feature_after_dropout4 = self.dropout4(visual_feature)
        visual_feature_after_dropout5 = self.dropout5(visual_feature)
        contextual_feature1 = self.SequenceModeling(visual_feature_after_dropout1)
        contextual_feature2 = self.SequenceModeling(visual_feature_after_dropout2)
        contextual_feature3 = self.SequenceModeling(visual_feature_after_dropout3)
        contextual_feature4 = self.SequenceModeling(visual_feature_after_dropout4)
        contextual_feature5 = self.SequenceModeling(visual_feature_after_dropout5)
        contextual_feature =  ( (contextual_feature1).add ((contextual_feature2).add(((contextual_feature3).add(((contextual_feature4).add(contextual_feature5)))))) ) * (1/5)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())
        return prediction
