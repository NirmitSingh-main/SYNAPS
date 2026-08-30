import torch
import torch.nn as nn


class SignalTransformer(nn.Module):
    """
    Transformer model for signal classification.

    Expected input:
        x: Tensor of shape
           (batch_size, sequence_length, input_features)

    Example:
        input_features = 4
        sequence_length = 2000

    Output:
        logits of shape
        (batch_size, num_classes)
    """

    def __init__(
        self,
        input_features=4,
        num_classes=4,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_sequence_length=20000,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead"
            )

        self.input_features = input_features
        self.num_classes = num_classes
        self.d_model = d_model

        # Convert signal features into Transformer embedding
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
            x:
                Tensor with shape:
                (batch, sequence_length, input_features)

        Returns:
            logits:
                Tensor with shape:
                (batch, num_classes)
        """

        if x.ndim != 3:
            raise ValueError(
                "Input must have shape "
                "(batch, sequence_length, input_features)"
            )

        if x.shape[-1] != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} input features, "
                f"got {x.shape[-1]}"
            )

        sequence_length = x.shape[1]

        if sequence_length > self.positional_embedding.shape[1]:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum supported length "
                f"{self.positional_embedding.shape[1]}"
            )

        # Feature → embedding
        x = self.input_projection(x)

        # Add positional information
        x = x + self.positional_embedding[:, :sequence_length, :]

        # Transformer encoder
        x = self.encoder(x)

        # Normalize
        x = self.norm(x)

        # Global average pooling over sequence
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits