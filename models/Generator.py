"""Generador SRGAN-CKAN con perfil de grados por bloque residual."""
import math
from collections.abc import Sequence

from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from models.ResidualBlock import ResidualBlock


class Generator(nn.Module):
    """Generador SRGAN con d_l variable en cada bloque residual.

    spline_degrees=None: todos los bloques usan external_spline_degree.
    spline_degrees=int: grado uniforme explícito.
    spline_degrees=secuencia: un grado por bloque residual.

    Las capas externas usan external_spline_degree, independientemente
    del perfil residual, para aislar su efecto en los experimentos.
    """

    def __init__(self, num_residual_blocks=16, scale_factor=4, norm: str = 'in',
                 post_activation: bool = True, spline_degrees=None,
                 external_spline_degree: int = 3, grid_size: int = 8,
                 projector_type: str = 'spline', chunk_pixels: int = 8192,
                 projector_chunk_size: int = 256):
        super().__init__()
        if scale_factor not in (2, 4, 8):
            raise ValueError('scale_factor debe ser 2, 4 u 8')
        if isinstance(num_residual_blocks, bool) or not isinstance(num_residual_blocks, int) or num_residual_blocks < 1:
            raise ValueError('num_residual_blocks debe ser un entero positivo')
        if isinstance(external_spline_degree, bool) or not isinstance(external_spline_degree, int) or external_spline_degree < 1:
            raise ValueError('external_spline_degree debe ser un entero positivo')

        if spline_degrees is None:
            profile = (external_spline_degree,) * num_residual_blocks
        elif isinstance(spline_degrees, int) and not isinstance(spline_degrees, bool):
            profile = (spline_degrees,) * num_residual_blocks
        elif isinstance(spline_degrees, Sequence) and not isinstance(spline_degrees, (str, bytes)):
            profile = tuple(spline_degrees)
        else:
            raise ValueError('spline_degrees debe ser None, entero o secuencia de enteros')
        if len(profile) != num_residual_blocks:
            raise ValueError(f'Se esperaban {num_residual_blocks} grados, se recibieron {len(profile)}')
        if any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in profile):
            raise ValueError('Cada grado debe ser un entero positivo')
        self.spline_degrees = profile
        self.external_spline_degree = external_spline_degree
        self.projector_type = projector_type

        def conv(cin, cout, kernel, padding, degree):
            return nn.KAN_Conv2d(
                cin, cout, kernel_size=kernel, padding=padding,
                grid_size=grid_size, spline_degree=degree,
                projector_type=projector_type, chunk_pixels=chunk_pixels,
                projector_chunk_size=projector_chunk_size)

        def norm_layer(kind, c):
            if kind == 'bn':
                return nn.BatchNorm2d(c, affine=True)
            if kind == 'in':
                return nn.InstanceNorm2d(c, affine=True, track_running_stats=False)
            if kind == 'none':
                return nn.Identity()
            raise ValueError(f'norm inválido: {kind}')

        self.initial = nn.Sequential(conv(3, 64, 9, 4, external_spline_degree), nn.PReLU())
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(64, norm=norm, post_activation=post_activation,
                          spline_degree=d, grid_size=grid_size,
                          projector_type=projector_type, chunk_pixels=chunk_pixels,
                          projector_chunk_size=projector_chunk_size)
            for d in profile
        ])
        self.post_res = nn.Sequential(conv(64, 64, 3, 1, external_spline_degree),
                                      norm_layer(norm, 64))
        up_layers = []
        for _ in range(int(math.log2(scale_factor))):
            up_layers += [conv(64, 256, 3, 1, external_spline_degree),
                          nn.PixelShuffle(2), nn.PReLU()]
        self.upsample = nn.Sequential(*up_layers)
        self.final = nn.Sequential(conv(64, 3, 9, 4, external_spline_degree), nn.Tanh())
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.KAN_Conv2d):
                continue
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
            # No reinicializar nn.Linear de los proyectores spline.

    def forward(self, x):
        initial = self.initial(x)
        segments = max(1, min(4, len(self.res_blocks)))
        x = checkpoint_sequential(self.res_blocks, segments=segments,
                                  input=initial, use_reentrant=False)
        x = self.post_res(x) + initial
        return self.final(self.upsample(x))
