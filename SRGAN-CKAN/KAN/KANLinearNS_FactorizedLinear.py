import math

import torch
from torch import nn

EPS = 1e-12

class KANLinearNS_FactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, p=2, s=4, base_activation=nn.SiLU):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.D = grid_size + spline_order  # n + m
        self.p = p
        self.s = s

        self.base_activation = base_activation()

        # Coeficientes entrenables a_{jk} ∈ ℝ
        self.a = nn.Parameter(torch.randn(p, s))

        # Matrices fijas M_{jk} ∈ ℝ^{in × out}
        self.register_buffer("M", torch.randn(p, s, in_features, out_features))

        # Pesos spline y malla
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, self.D))
        self.register_buffer("grid", torch.linspace(-1, 1, grid_size + 2 * spline_order + 1))

        self.norm = nn.LayerNorm(out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.a.normal_(mean=0.0, std=0.1)

    def factorized_linear_sum(self, x: torch.Tensor) -> torch.Tensor:
        """
        W = ∑_j ∑_k a_{jk} M_{jk}  ∈ ℝ^{in × out}
        """
        W = torch.zeros(self.in_features, self.out_features, device=x.device, dtype=x.dtype)
        M = self.M.to(device=x.device, dtype=x.dtype)  # mantener compatibilidad sin cambiar complejidad
        for j in range(self.p):
            for k in range(self.s):
                W = W + self.a[j, k] * M[j, k]
        return x @ W  # (B, in) @ (in, out) -> (B, out)

    def forward(self, x: torch.Tensor):
        x_activated = self.base_activation(x)
        base_output = self.factorized_linear_sum(x_activated)  # (batch, out)
        spline_output = torch.einsum('bi,oij->bo', x, self.spline_weight)
        return self.norm(base_output + spline_output)

    @staticmethod
    def compute_new_coeffs(n: int, m: int, knots: torch.Tensor, coeffs: torch.Tensor):
        """
        coeffs: (n, n+m, m+1)
        Traducción directa con salvaguardas de índices y divisiones (sin cambiar bucles).
        """
        # Bloque 1
        for j in range(n):
            if j + m + 1 < knots.numel():
                numerator = (knots[j + 1 + m] - knots[j + m]).clamp_min(EPS) ** (m - 1)
                # coeffs[j, j+m, m]
                val_m = numerator.clone()
                for k in range(2, m + 1):
                    if j + k + m < knots.numel():
                        den = (knots[j + k + m] - knots[j + m]).clamp_min(EPS)
                        val_m = val_m / den
                coeffs[j, j + m, m] = val_m

                # coeffs[j, j, 0]
                val_0 = numerator.clone()
                for k in range(2, m + 1):
                    idx = j + 1 - k + m
                    if 0 <= idx < knots.numel():
                        den = (knots[j + 1 + m] - knots[idx]).clamp_min(EPS)
                        val_0 = val_0 / den
                coeffs[j, j, 0] = val_0

        # Bloque 2
        for i in range(n - 2, n - m - 1, -1):
            if (i + m) < knots.numel() and (n + m) < knots.numel():
                c1 = (knots[n - 1 + m] - knots[i + m]) / (knots[n + m] - knots[i + m] + EPS)
                c2 = (knots[n + m] - knots[n - 1 + m]) / (knots[n + m] - knots[i + 1 + m] + EPS)
                for k in range(m - 1, -1, -1):
                    coeffs[n - 1, i + m, k] = (
                        c1 * coeffs[n - 1, i + m, k + 1] +
                        c2 * coeffs[n - 1, i + 1 + m, k + 1]
                    )

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """
        Mantiene la misma complejidad: recalcula nudos y coeficientes y copia
        solo la parte con forma compatible (sin operaciones extra).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        # Actualizar nudos al rango global con margen
        xmin = x.min().item()
        xmax = x.max().item()
        span = max(xmax - xmin, 1e-3)
        lo = xmin - margin * span
        hi = xmax + margin * span

        new_grid = torch.linspace(lo, hi, self.grid_size + 2 * self.spline_order + 1,
                                  device=self.grid.device, dtype=self.grid.dtype)
        self.grid.copy_(new_grid)

        # Recalcular coeficientes y copiar bloque compatible
        n, m = self.grid_size, self.spline_order
        knots = self.grid.to(device=x.device, dtype=torch.float32)
        coeffs = torch.zeros(n, n + m, m + 1, dtype=torch.float32, device=x.device)
        self.compute_new_coeffs(n, m, knots, coeffs)

        o_fill = min(self.out_features, n)
        i_fill = min(self.in_features, n + m)
        d_fill = min(self.D, m + 1)

        if o_fill > 0 and i_fill > 0 and d_fill > 0:
            self.spline_weight.data[:o_fill, :i_fill, :d_fill].copy_(
                coeffs[:o_fill, :i_fill, :d_fill].to(self.spline_weight.dtype)
            )