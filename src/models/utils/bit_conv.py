import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def activation_quant(x: Tensor, dim: int = 1):
    """
    Per-channel quantization to 8 bits.

    Args:
        x (Tensor): Input tensor.
        dim (int): Dimension along which to compute the scale.

    Returns:
        Tensor: Quantized tensor.
    """
    scale = 127.0 / x.abs().amax(dim=dim, keepdim=True).clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w: Tensor, with_scale = False):
    """
    Quantizes weights to binary values.

    Args:
        w (Tensor): Weight tensor.

    Returns:
        Tensor: Quantized weight tensor.
    """
    e = w.mean()
    if with_scale == True:
        scale = w.abs().mean()
        u = (w - e).sign() * scale
    else:
        u = (w - e).sign()
    return u

class SimpleRMSNorm(nn.Module):
    """
    Simple RMS normalization module.

    Args:
        dim (int): Dimension along which to compute the RMS.
        eps (float): A small epsilon for numerical stability.
    """
    def __init__(self, dim: int = 1, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor):
        """
        Forward pass for RMS normalization.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Normalized tensor.
        """
        rms = x.pow(2).mean(dim=self.dim, keepdim=True).sqrt() + self.eps
        x_norm = x / rms
        return x_norm

class BitConv1d(nn.Conv1d):
    """
    1D convolutional layer with bit quantization.

    Inherits from nn.Conv1d and applies quantization to activations and weights.
    """
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = SimpleRMSNorm(dim=1)(x)

        # Straight-Through Estimator (STE) using detach
        x_quant = x_norm + (activation_quant(x_norm, dim=1) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.conv1d(x_quant, w_quant, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        return y

class BitConv2d(nn.Conv2d):
    """
    2D convolutional layer with bit quantization.

    Inherits from nn.Conv2d and applies quantization to activations and weights.
    """
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = SimpleRMSNorm(dim=1)(x)

        # Straight-Through Estimator (STE) using detach
        x_quant = x_norm + (activation_quant(x_norm, dim=1) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.conv2d(x_quant, w_quant, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        return y

class BitConv3d(nn.Conv3d):
    """
    3D convolutional layer with bit quantization.

    Inherits from nn.Conv3d and applies quantization to activations and weights.
    """
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x = x.to(self.weight.dtype)
        x_norm = SimpleRMSNorm(dim=1)(x)

        # Straight-Through Estimator (STE) using detach
        x_quant = x_norm + (activation_quant(x_norm, dim=1) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.conv3d(x_quant, w_quant, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        return y