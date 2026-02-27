#!/usr/bin/python
# -*- coding:utf-8 -*-

import warnings  # Import the warning prompt module

from torch import nn  # Import the neural network module of PyTorch


# --------------Adapted for time-series input with length 1024 + multi-channel (3/6) + 7-class output-------------------
class CNN(nn.Module):
    """
    1-dimensional Convolutional Neural Network (1D CNN) model
    Application scenario: Process time-series data with length 1024, support 3/6 channel input, and output 7-class classification results
    """

    def __init__(self, pretrained=False, in_channel=3, out_channel=7):
        """
        Model initialization function
        Args:
            pretrained (bool): Whether to load pre-trained weights
            in_channel (int): Number of channels of input data, supporting 3 or 6 (default: 3)
            out_channel (int): Number of output classes, default: 7 classes
        """
        super(CNN, self).__init__()  # Inherit the initialization method of nn.Module

        # Note: The current model has no pre-trained weights, a warning will be thrown if pretrained=True is set
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        # Input tensor shape: (batch_size, in_channel, 1024) → batch_size=batch size, in_channel=3/6, 1024=time-series length
        # First convolutional layer: Extract basic time-series features
        self.layer1 = nn.Sequential(
            # 1D convolution: Input channels in_channel, output channels 16, kernel size 15 (time-series dimension)
            # Dimension change: (batch, in_channel, 1024) → (batch, 16, 1024-15+1) = (batch,16,1010)
            nn.Conv1d(in_channel, 16, kernel_size=15),
            nn.BatchNorm1d(16),  # Batch normalization: Accelerate training and prevent gradient vanishing
            nn.ReLU(inplace=True))  # Activation function: Introduce non-linearity, inplace=True saves memory

        # Second convolutional layer + pooling: Deepen feature extraction and reduce dimensions
        self.layer2 = nn.Sequential(
            # 1D convolution: Input channels 16, output channels 32, kernel size 3
            # Dimension change: (batch,16,1010) → (batch,32, 1010-3+1) = (batch,32,1008)
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # 1D max pooling: Pooling kernel size 2, stride 2, reduce time-series dimension
            # Dimension change: (batch,32,1008) → (batch,32, 1008//2) = (batch,32,504)
            nn.MaxPool1d(kernel_size=2, stride=2))

        # Third convolutional layer: Further extract high-level time-series features
        self.layer3 = nn.Sequential(
            # 1D convolution: Input channels 32, output channels 64, kernel size 3
            # Dimension change: (batch,32,504) → (batch,64, 504-3+1) = (batch,64,502)
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        # Fourth convolutional layer + adaptive pooling: Extract deep features and fix output dimension (key)
        self.layer4 = nn.Sequential(
            # 1D convolution: Input channels 64, output channels 128, kernel size 3
            # Dimension change: (batch,64,502) → (batch,128, 502-3+1) = (batch,128,500)
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # Adaptive max pooling: Regardless of the input time-series dimension, the output is fixed to 4
            # Dimension change: (batch,128,500) → (batch,128,4), compatible with input of different lengths
            nn.AdaptiveMaxPool1d(4))

        # Fully connected layer: Map convolutional features to classification space
        self.layer5 = nn.Sequential(
            # Linear layer: Input dimension 128*4 (after flattening convolutional features), output dimension 256
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            # Linear layer: Dimensionality reduction, input 256, output 64
            nn.Linear(256, 64),
            nn.ReLU(inplace=True))

        # Final classification layer: Input 64-dimensional features, output out_channel classes (default: 7 classes)
        self.fc = nn.Linear(64, out_channel)

    def forward(self, x):
        """
        Model forward propagation function: Define the path of data flowing through each layer
        Args:
            x (tensor): Input tensor with shape (batch_size, in_channel, 1024)
        Returns:
            tensor: Output tensor with shape (batch_size, out_channel), corresponding to the prediction scores of 7 classes
        """
        # x: (batch_size, in_channel, 1024) → Multi-channel time-series input
        x = self.layer1(x)  # First convolutional layer: Extract basic features
        x = self.layer2(x)  # Second convolutional layer + pooling: Deepen features + reduce dimensions
        x = self.layer3(x)  # Third convolutional layer: High-level feature extraction
        x = self.layer4(x)  # Fourth convolutional layer + adaptive pooling: Fix feature dimension

        # Feature flattening: Convert 3D tensor (batch, 128, 4) to 2D tensor (batch, 128*4=512) to adapt to fully connected layers
        x = x.view(x.size(0), -1)

        x = self.layer5(x)  # Fully connected layer: Feature dimensionality reduction
        x = self.fc(x)  # Classification layer: Output prediction scores for 7 classes

        return x
