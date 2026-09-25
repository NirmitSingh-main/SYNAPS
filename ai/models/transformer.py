import torch
import torch.nn as nn


class IQConvFrontEnd(nn.Module):
    """
    Multi-scale strided 1D Convolutional Front-End for raw IQ waveforms.
    Transforms raw input channels into 256 learned token representations.

    Replaces the uniform-kernel (k=7) front-end with a hierarchical strided design
    that dramatically expands the receptive field (k=31 in layer 1, spanning multiple
    symbols across all samples-per-symbol rates) so the model can resolve amplitude
    transitions and constellation geometry between QPSK and QAM16.

    Input:
        (batch, in_channels, N)
        - in_channels=2: [I, Q]
        - in_channels=3: [I, Q, power]
        - in_channels=5: [I, Q, mag, cos_dphi, sin_dphi]
    Output:
        (batch, 256, out_channels)
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 64, num_tokens: int = 256):
        super().__init__()

        self.conv = nn.Sequential(
            # Layer 1: Multi-symbol receptive field (kernel 31 spans 1.5 - 6.2 symbols across all sps)
            nn.Conv1d(in_channels, 32, kernel_size=31, stride=4, padding=15),
            nn.BatchNorm1d(32),
            nn.GELU(),
            # Layer 2: Intermediate symbol-transition hierarchy
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            # Layer 3: Deep constellation feature representation
            nn.Conv1d(64, out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: Tensor with shape (batch, in_channels, N)
        Returns:
            Tensor with shape (batch, num_tokens, out_channels)
        """
        h = self.conv(x)        # (batch, out_channels, N')
        h = self.pool(h)        # (batch, out_channels, num_tokens)
        h = h.transpose(1, 2)   # (batch, num_tokens, out_channels)
        return h


class SignalTransformer(nn.Module):
    """
    Transformer model for signal classification.

    Supports dual input modes:
    1. Raw IQ waveforms: (batch, C, N) via learned Conv1D front-end
       C=2: [I, Q]
       C=3: [I, Q, I^2+Q^2]
       C=5: [I, Q, mag, cos_dphi, sin_dphi]
    2. Handcrafted tokens: (batch, sequence_length, input_features) via linear projection

    Output:
        logits of shape (batch, num_classes)
    """

    def __init__(
        self,
        input_features=5,
        num_classes=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_sequence_length=20000,
        use_conv_frontend=None,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead"
            )

        self.input_features = input_features
        self.num_classes = num_classes
        self.d_model = d_model

        if use_conv_frontend is None:
            self.use_conv_frontend = (input_features in (2, 3, 5))
        else:
            self.use_conv_frontend = use_conv_frontend

        if self.use_conv_frontend:
            conv_in_channels = input_features if input_features in (2, 3, 5) else 5
            self.conv_frontend = IQConvFrontEnd(
                in_channels=conv_in_channels,
                out_channels=d_model,
                num_tokens=256,
            )
            self.input_projection = None
        else:
            self.conv_frontend = None
            self.input_projection = nn.Linear(
                input_features,
                d_model
            )

        # Learnable positional encoding
        self.positional_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_sequence_length,
                d_model
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters:
            x: Tensor of shape:
               - (batch, C, num_samples) if use_conv_frontend is True
               - (batch, sequence_length, input_features) if use_conv_frontend is False

        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        if self.use_conv_frontend:
            expected_c = self.conv_frontend.conv[0].in_channels
            if x.ndim == 3 and x.shape[1] != expected_c and x.shape[2] == expected_c:
                x = x.transpose(1, 2)
            if x.ndim != 3 or x.shape[1] != expected_c:
                raise ValueError(
                    f"Conv1D front-end expects input (batch, {expected_c}, num_samples), got {x.shape}"
                )
            x = self.conv_frontend(x)  # (batch, 256, d_model)
        else:
            if x.ndim != 3:
                raise ValueError(
                    "Input must have shape (batch, sequence_length, input_features)"
                )
            if x.shape[-1] != self.input_features:
                raise ValueError(
                    f"Expected {self.input_features} input features, got {x.shape[-1]}"
                )
            x = self.input_projection(x)

        sequence_length = x.shape[1]

        if sequence_length > self.positional_embedding.shape[1]:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum supported length {self.positional_embedding.shape[1]}"
            )

        # Add positional information
        x = x + self.positional_embedding[:, :sequence_length, :]

        # Transformer encoder
        x = self.encoder(x)

        # Normalize
        x = self.norm(x)

        # Global average pooling over token sequence (restored baseline)
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits