"""Convolución CKAN: im2col + operador funcional B-spline.

Para cada posición espacial p, F.unfold produce x_p en R^(C_in*kH*kW).
El proyector implementa
    y_o(p) = sum_i W_base[o,i] SiLU(x_p[i])
           + sum_i sum_r C[o,i,r] B_(r,d)(x_p[i]) + b[o].

El grado d es configurable por instancia (futuro: por bloque residual).
La ruta por bandas evita materializar todos los parches de una imagen.
NOTA: no equivale a Conv2d ordinaria, salvo al usar el proyector lineal.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from KAN.KANSplineProjector import KANSplineProjector
from KAN._KANProjector import _KANProjector


class KAN_Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3,
                 stride=1, padding=0, dilation=1, bias=True,
                 chunk_pixels: int = 8192, *, grid_size: int = 8,
                 spline_degree: int = 3, grid_range=(-1.0, 1.0),
                 projector_chunk_size: int = 256,
                 projector_type: str = 'spline'):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        if min(in_channels, out_channels, chunk_pixels) <= 0:
            raise ValueError('Canales y chunk_pixels deben ser positivos')
        if any(v <= 0 for v in (*self.kernel_size, *self.stride, *self.dilation)):
            raise ValueError('kernel_size, stride y dilation deben ser positivos')
        if any(v < 0 for v in self.padding):
            raise ValueError('padding no puede ser negativo')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.chunk_pixels = int(chunk_pixels)
        self.projector_type = projector_type
        in_features = in_channels * self.kernel_size[0] * self.kernel_size[1]
        if projector_type == 'spline':
            self.KAN_NM = KANSplineProjector(
                in_features, out_channels, grid_size=grid_size,
                spline_degree=spline_degree, grid_range=grid_range,
                bias=bias, chunk_size=projector_chunk_size)
        elif projector_type == 'linear':
            self.KAN_NM = _KANProjector(in_features, out_channels, bias=bias)
        else:
            raise ValueError("projector_type debe ser 'spline' o 'linear'")

    @staticmethod
    def _out_dim(n, p, d, k, s):
        return (n + 2*p - d*(k-1) - 1)//s + 1

    def forward(self, x: torch.Tensor, update_grid: bool = False):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError('Se espera (B, in_channels, H, W)')
        B, _, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        ho = self._out_dim(H, ph, dh, kh, sh)
        wo = self._out_dim(W, pw, dw, kw, sw)
        if ho <= 0 or wo <= 0:
            raise ValueError('La configuración produce una salida espacial vacía')

        if ho*wo <= self.chunk_pixels:
            patches = F.unfold(x, self.kernel_size, dilation=self.dilation,
                               padding=self.padding, stride=self.stride)
            return self.KAN_NM(patches, update_grid=update_grid).reshape(
                B, self.out_channels, ho, wo)

        # Cada banda cubre exactamente [row, end) filas de salida.
        # En vez de hacer padding al resultado, construimos el campo
        # receptivo exacto con padding vertical explícito.
        output_bands = []
        rows_per_band = max(1, self.chunk_pixels // wo)
        for row in range(0, ho, rows_per_band):
            end = min(row + rows_per_band, ho)
            top = row*sh - ph
            bottom = (end-1)*sh - ph + (kh-1)*dh + 1
            crop_top, crop_bottom = max(0, top), min(H, bottom)
            crop = x[:, :, crop_top:crop_bottom, :]
            crop = F.pad(crop, (0, 0, max(0, -top), max(0, bottom-H)))
            patches = F.unfold(crop, self.kernel_size, dilation=self.dilation,
                               padding=(0, pw), stride=self.stride)
            expected = (end-row)*wo
            if patches.shape[-1] != expected:
                raise RuntimeError(f'Banda incorrecta: {patches.shape[-1]} != {expected}')
            projected = self.KAN_NM(patches, update_grid=update_grid)
            output_bands.append(projected.reshape(B, self.out_channels, end-row, wo))
        return torch.cat(output_bands, dim=2)


# Mantiene compatibilidad con los modelos que usan nn.KAN_Conv2d.
setattr(nn, 'KAN_Conv2d', KAN_Conv2d)
