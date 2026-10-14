"""Bases B-spline uniformes abiertas para operadores CKAN.

Correspondencia teórica:
    B_{i,d}(x): i-ésima función base B-spline de grado d.
    t_i: nudos del vector no decreciente de nudos.
    N = grid_size + degree: número de funciones base.

Este módulo calcula las bases, NO los coeficientes entrenables c_i de
phi_d(x) = sum_i c_i B_{i,d}(x). Esos pertenecen al proyector CKAN.
"""

import torch
from torch import nn


class BSplineBasis(nn.Module):
    """Evalúa B_{i,d}(x) mediante la recurrencia de Cox-de Boor.

    Args:
        grid_size: número de intervalos uniformes en el dominio.
        degree: grado polinomial d (no confundir con orden d+1).
        grid_range: extremos [a,b] de la malla.

    Entrada: tensor x de forma arbitraria (...).
    Salida: tensor (..., grid_size + degree), una base por última dimensión.

    La malla se registra como buffer para que siga al módulo entre CPU/CUDA
    y se almacene en state_dict, pero no sea un parámetro aprendible.
    """

    def __init__(self, grid_size: int = 8, degree: int = 3,
                 grid_range: tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        if not isinstance(grid_size, int) or grid_size < 1:
            raise ValueError("grid_size debe ser un entero >= 1")
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree debe ser un entero >= 1")
        a, b = map(float, grid_range)
        if not a < b:
            raise ValueError("grid_range debe satisfacer a < b")

        self.grid_size = grid_size
        self.degree = degree
        self.grid_range = (a, b)
        self.n_basis = grid_size + degree

        # Vector abierto: d+1 repeticiones en cada extremo.
        # Longitud = N + d + 1 = grid_size + 2*d + 1.
        interior = torch.linspace(a, b, grid_size + 1, dtype=torch.float64)[1:-1]
        knots = torch.cat((
            torch.full((degree + 1,), a, dtype=torch.float64),
            interior,
            torch.full((degree + 1,), b, dtype=torch.float64),
        ))
        self.register_buffer("knots", knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError("x debe ser un tensor de punto flotante")

        # Se evalúan las bases en FP32 (o FP64) para evitar inestabilidad
        # numérica en la recurrencia bajo autocast/AMP.
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        z = x.to(dtype=work_dtype)
        t = self.knots.to(dtype=work_dtype)
        a, b = self.grid_range

        # Caso base de Cox-de Boor:
        # B_{i,0}(x) = 1 si t_i <= x < t_{i+1}, y 0 en otro caso.
        # La convención de extremo derecho x=b se trata al finalizar.
        z_exp = z.unsqueeze(-1)
        basis = ((z_exp >= t[:-1]) & (z_exp < t[1:])).to(work_dtype)

        # Recurrencia para elevar el grado desde 0 hasta d:
        # B_{i,k} = [(x-t_i)/(t_{i+k}-t_i)] B_{i,k-1}
        #         + [(t_{i+k+1}-x)/(t_{i+k+1}-t_{i+1})] B_{i+1,k-1}.
        # Si un denominador es cero, su término se define como cero.
        for k in range(1, self.degree + 1):
            count = t.numel() - k - 1
            left_den = t[k:k + count] - t[:count]
            right_den = t[k + 1:k + 1 + count] - t[1:1 + count]

            left_weight = torch.where(
                left_den > 0,
                (z_exp - t[:count]) / left_den.clamp_min(torch.finfo(work_dtype).tiny),
                torch.zeros_like(z_exp - t[:count]),
            )
            right_weight = torch.where(
                right_den > 0,
                (t[k + 1:k + 1 + count] - z_exp) /
                right_den.clamp_min(torch.finfo(work_dtype).tiny),
                torch.zeros_like(z_exp - t[:count]),
            )
            basis = left_weight * basis[..., :count] + right_weight * basis[..., 1:count + 1]

        # En la malla abierta, la última base vale 1 exactamente en x=b.
        # Fuera de [a,b], todas las bases se mantienen en cero.
        at_right = z == b
        last_basis = torch.nn.functional.one_hot(
            torch.full_like(z, self.n_basis - 1, dtype=torch.long),
            num_classes=self.n_basis,
        ).to(work_dtype)
        basis = torch.where(at_right.unsqueeze(-1), last_basis, basis)
        return basis
