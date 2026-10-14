"""Bloque residual SR con grado B-spline configurable por bloque."""
from torch import nn


class ResidualBlock(nn.Module):
    """Dos convoluciones CKAN de grado d_l con conexión residual.

    La pareja de operadores comparte el grado polinomial, pero NO sus
    coeficientes entrenables. Para G intervalos, cada operador spline tiene
    C_out * (C_in * kH * kW) * (G + d_l) coeficientes.
    """

    def __init__(self, channels, norm: str = 'in', post_activation: bool = True,
                 spline_degree: int = 3, grid_size: int = 8,
                 projector_type: str = 'spline', chunk_pixels: int = 8192,
                 projector_chunk_size: int = 256):
        super().__init__()
        if isinstance(spline_degree, bool) or not isinstance(spline_degree, int) or spline_degree < 1:
            raise ValueError('spline_degree debe ser un entero positivo')
        self.spline_degree = spline_degree
        self.projector_type = projector_type

        def norm_layer(kind, c):
            if kind == 'bn':
                return nn.BatchNorm2d(c, affine=True)
            if kind == 'in':
                return nn.InstanceNorm2d(c, affine=True, track_running_stats=False)
            if kind == 'none':
                return nn.Identity()
            raise ValueError(f'norm inválido: {kind}')

        def conv():
            return nn.KAN_Conv2d(
                channels, channels, kernel_size=3, padding=1,
                spline_degree=spline_degree, grid_size=grid_size,
                projector_type=projector_type, chunk_pixels=chunk_pixels,
                projector_chunk_size=projector_chunk_size)

        self.block = nn.Sequential(conv(), norm_layer(norm, channels), nn.PReLU(),
                                   conv(), norm_layer(norm, channels))
        self.post_act = nn.PReLU() if post_activation else nn.Identity()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.KAN_Conv2d):
                continue  # No reinicializar los coeficientes spline.
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
            # No tocar nn.Linear interno de KANSplineProjector.

    def forward(self, x):
        return self.post_act(x + self.block(x))
