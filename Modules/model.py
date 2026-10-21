"""
model.py
--------
ECG 1D CNN-BiLSTM architecture for single-lead arrhythmia classification.

Architecture:
  Input: (B, 1, 2500)  — Batch x 1 channel x 2500 samples @ 250 Hz
  Stage 1 — 1D CNN Feature Extractor (4 ConvBlocks)
  Stage 2 — Bidirectional LSTM Temporal Modelling
  Stage 3 — Classifier Head → (B, NUM_CLASSES)

Reference: docs/Architecture.md, docs/Design.md
"""

import torch
import torch.nn as nn
from torch import Tensor

from config import (
    NUM_CLASSES, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_HIDDEN_SIZE, CLASSIFIER_DROPOUT,
)


class ConvBlock(nn.Module):
    """A single 1D convolutional block: Conv1d → BatchNorm1d → ReLU → MaxPool1d.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output (filter) channels.
    kernel_size : int
        Conv1d kernel size.
    stride : int
        Conv1d stride. Default 1.
    padding : int
        Conv1d zero-padding. Default 0.
    pool_size : int
        MaxPool1d pool size. Default 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        pool_size: int = 2,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, in_channels, L)

        Returns
        -------
        Tensor, shape (B, out_channels, L // (stride * pool_size))
        """
        return self.block(x)


class ECGCNNBiLSTM(nn.Module):
    """1D CNN + Bidirectional LSTM hybrid model for ECG arrhythmia classification.

    Input shape : (B, 1, 2500) — single channel, 2500 samples (10 sec @ 250 Hz)
    Output shape: (B, NUM_CLASSES) — raw logits (no softmax; use with CrossEntropyLoss)

    Stage 1 — CNN Feature Extractor (sequence compression 2500 → 78):
      CB1: Conv1d(1,  32,  k=15, s=2, p=7)  + BN + ReLU + MaxPool1d(2) → L=625
      CB2: Conv1d(32, 64,  k=7,  s=1, p=3)  + BN + ReLU + MaxPool1d(2) → L=312
      CB3: Conv1d(64, 128, k=5,  s=1, p=2)  + BN + ReLU + MaxPool1d(2) → L=156
      CB4: Conv1d(128,128, k=3,  s=1, p=1)  + BN + ReLU + MaxPool1d(2) → L=78

    Stage 2 — BiLSTM Temporal Modelling:
      Permute (B,128,78) → (B,78,128)
      LSTM(input=128, hidden=128, layers=2, bidirectional=True, dropout=0.3)
      Global Average Pooling over time → (B, 256)

    Stage 3 — Classifier Head:
      Linear(256, 64) → ReLU → Dropout(0.3) → Linear(64, NUM_CLASSES)

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes. Defaults to config.NUM_CLASSES (5).
    lstm_hidden : int, optional
        LSTM hidden state size per direction. Defaults to config.LSTM_HIDDEN_SIZE (128).
    lstm_layers : int, optional
        Number of LSTM layers. Defaults to config.LSTM_NUM_LAYERS (2).
    lstm_dropout : float, optional
        Dropout between LSTM layers. Defaults to config.LSTM_DROPOUT (0.3).
    fc_hidden : int, optional
        FC hidden layer size. Defaults to config.FC_HIDDEN_SIZE (64).
    clf_dropout : float, optional
        Dropout in classifier head. Defaults to config.CLASSIFIER_DROPOUT (0.3).
    """

    def __init__(
        self,
        num_classes:  int   = NUM_CLASSES,
        lstm_hidden:  int   = LSTM_HIDDEN_SIZE,
        lstm_layers:  int   = LSTM_NUM_LAYERS,
        lstm_dropout: float = LSTM_DROPOUT,
        fc_hidden:    int   = FC_HIDDEN_SIZE,
        clf_dropout:  float = CLASSIFIER_DROPOUT,
    ):
        super().__init__()

        # ---- Stage 1: CNN feature extractor ----
        self.cnn = nn.Sequential(
            ConvBlock(1,   32,  kernel_size=15, stride=2, padding=7),   # 2500 → 625
            ConvBlock(32,  64,  kernel_size=7,  stride=1, padding=3),   # 625  → 312
            ConvBlock(64,  128, kernel_size=5,  stride=1, padding=2),   # 312  → 156
            ConvBlock(128, 128, kernel_size=3,  stride=1, padding=1),   # 156  → 78
        )

        cnn_out_channels = 128   # output channels from last ConvBlock
        # seq_len after CNN = 78 (verified in N4 notebook)

        # ---- Stage 2: BiLSTM temporal modelling ----
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2   # 256 (bidirectional)

        # ---- Stage 3: Classifier head ----
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(clf_dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, 1, 2500)
            Preprocessed ECG windows.

        Returns
        -------
        logits : Tensor, shape (B, num_classes)
            Raw class scores (use CrossEntropyLoss which applies log-softmax internally).
        """
        # Stage 1: CNN — (B,1,2500) → (B,128,78)
        cnn_out = self.cnn(x)

        # Stage 2: Permute for LSTM — (B,128,78) → (B,78,128)
        lstm_in = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)           # (B, 78, 256)

        # Global average pooling over time dimension
        pooled = lstm_out.mean(dim=1)              # (B, 256)

        # Stage 3: Classifier
        logits = self.classifier(pooled)           # (B, num_classes)
        return logits

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(device: str = "cpu") -> ECGCNNBiLSTM:
    """Convenience factory: construct and move model to device.

    Parameters
    ----------
    device : str
        PyTorch device string, e.g. 'cpu', 'cuda', 'cuda:0'.

    Returns
    -------
    ECGCNNBiLSTM
    """
    model = ECGCNNBiLSTM()
    model.to(device)
    return model
