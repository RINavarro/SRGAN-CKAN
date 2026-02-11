import torch
from torch import nn


class _KANProjector(nn.Module):
    """
    Proyector básico: (B, C*kH*kW, L) -> (B, out_channels, L)
    reemplaza internamente 'core' KAN (factorizado + splines).
    """
    def __init__(self, in_col_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        # Conv1d con kernel=1 sobre el eje de canales de im2col
        self.core = nn.Conv1d(in_col_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, col: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        # col: (B, C*kH*kW, L)
        return self.core(col)  # (B, out_channels, L)