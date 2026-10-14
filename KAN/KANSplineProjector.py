"""Proyector KAN basado en B-splines (Cox-de Boor).

Teoría: y_o = sum_i W_base[o,i] * SiLU(x_i)
              + sum_i sum_r C[o,i,r] * B_{r,d}(x_i) + b_o.

A diferencia de Conv1d(kernel_size=1), B_{r,d}(x_i) es una función
no lineal de cada entrada. El grado d controla el grado polinomial,
y N = grid_size + d controla el número de funciones base.

Este módulo NO actualiza nudos. La malla es fija para aislar el efecto
experimental del grado; update_grid=True produce un error explícito.
"""

import math
import torch
from torch import nn

from KAN.BSplineBasis import BSplineBasis


class KANSplineProjector(nn.Module):
    """Proyecta (B,I) -> (B,O) o (B,I,L) -> (B,O,L).

    Args:
        in_features: I, dimensiones de entrada (para im2col: C*kH*kW).
        out_features: O, dimensiones de salida.
        grid_size: G, número de intervalos del dominio spline.
        spline_degree: d, grado de B-spline (no orden d+1).
        grid_range: dominio cerrado [a,b]. Fuera de él la rama spline
            vale cero; la rama base sigue operando.
        chunk_size: número máximo de vectores de entrada evaluados por
            fragmento. Controla la memoria temporal de las bases (M,I,N).
    """

    def __init__(self, in_features: int, out_features: int,
                 grid_size: int = 8, spline_degree: int = 3,
                 grid_range: tuple[float, float] = (-1.0, 1.0),
                 bias: bool = True, chunk_size: int = 1024):
        super().__init__()
        if in_features < 1 or out_features < 1 or chunk_size < 1:
            raise ValueError('in_features, out_features y chunk_size deben ser positivos')
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.chunk_size = int(chunk_size)
        self.basis = BSplineBasis(grid_size=grid_size, degree=spline_degree,
                                 grid_range=grid_range)
        self.base_activation = nn.SiLU()
        self.base_linear = nn.Linear(in_features, out_features, bias=bias)

        # C[o,i,r] corresponde al coeficiente c_{oir} de la teoría.
        # No se comparte entre funciones base, canales de entrada ni salida.
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, self.basis.n_basis)
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.base_linear.reset_parameters()
        # Escala moderada: al aumentar N no debe crecer arbitrariamente
        # la amplitud inicial de la rama spline.
        nn.init.normal_(self.spline_weight, mean=0.0,
                        std=0.1 / math.sqrt(self.in_features))

    def _forward_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """x: (M,I). Acumula la salida por fragmentos sin materializar (M,I,N)."""
        outputs = []
        for block in x.split(self.chunk_size, dim=0):
            # La evaluación Cox-de Boor se realiza en FP32/FP64.
            bases = self.basis(block)  # (m,I,N)
            # Fuera de autocast para acumular la rama spline con precisión
            # de cálculo explícita; no altera el dtype del parámetro.
            with torch.autocast(device_type=block.device.type, enabled=False):
                spline = torch.einsum(
                    'min,oin->mo', bases,
                    self.spline_weight.to(dtype=bases.dtype)
                )
            base = self.base_linear(self.base_activation(block))
            # La salida mantiene la promoción de tipos normal de PyTorch.
            outputs.append(base + spline)
        return torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        if update_grid:
            raise NotImplementedError(
                'La actualización adaptativa de nudos requiere preservar '
                'la función spline; todavía no está implementada.'
            )
        if x.ndim not in (2, 3) or x.shape[1] != self.in_features:
            raise ValueError('Se espera x con forma (B,I) o (B,I,L), I=in_features')
        if x.ndim == 2:
            return self._forward_matrix(x)
        batch, _, length = x.shape
        # Cada posición espacial comparte los mismos coeficientes C[o,i,r].
        matrix = x.transpose(1, 2).reshape(-1, self.in_features)
        projected = self._forward_matrix(matrix)
        return projected.reshape(batch, length, self.out_features).transpose(1, 2)
