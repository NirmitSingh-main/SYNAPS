"""
Convolutional Neural Network (CNN) architectures for radio modulation classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignalCNN1D(nn.Module):
    """
    1D CNN for raw signal feature classification.
    Input shape: (batch_size, sequence_length, in_channels) or (batch_size, in_channels, sequence_length).
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        num_filters: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels, num_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(num_filters)

        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)

        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_filters * 4)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(num_filters * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is (batch, sequence_length, channels), transpose to (batch, channels, sequence_length)
        if x.ndim == 3 and x.shape[-1] == self.in_channels:
            x = x.transpose(1, 2)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


# Generic alias
SignalCNN = SignalCNN1D