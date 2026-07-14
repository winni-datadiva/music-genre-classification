"""
GenreCNN — CNN architecture for music genre classification.

Input: 2D MFCC arrays, shape (batch, 1, 13, 130)
        (1 channel, 13 MFCC coefficients, 130 time frames)
Output: logits over num_classes genres

Designed to be imported standalone:
    from model import GenreCNN
    model = GenreCNN(num_classes=len(GENRES))
"""

import torch
import torch.nn as nn


class GenreCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        # Block 1: (1, 13, 130) -> (16, 6, 65)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Block 2: (16, 6, 65) -> (32, 3, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Block 3: (32, 3, 32) -> (64, 1, 16)
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Collapse spatial dims regardless of exact size going in,
        # so this still works if TARGET_FRAMES or N_MFCC change later.
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 13, 130)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)     # (batch, 64, 1, 1)
        x = self.classifier(x)      # (batch, num_classes)
        return x


if __name__ == "__main__":
    # Quick shape sanity check — matches your dataset's confirmed batch shape
    # from Cell 25: (32, 1, 13, 130)
    model = GenreCNN(num_classes=10)
    dummy = torch.randn(32, 1, 13, 130)
    out = model(dummy)
    print("Input shape: ", dummy.shape)
    print("Output shape:", out.shape)  # expect (32, 10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
