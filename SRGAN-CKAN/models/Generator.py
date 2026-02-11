import math

from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from models.ResidualBlock import ResidualBlock


class Generator(nn.Module):
    """Generador SRGAN con KAN_Conv2d.
    Cambios:
    - Llama a _initialize_weights()
    - Normalización consistente con ResidualBlock (parametrizable)
    - checkpoint_sequential seguro para n_bloques pequeños
    - Upsample dinámico según scale_factor (potencias de 2)
    """

    def __init__(self, num_residual_blocks=16, scale_factor=4, norm: str = 'in', post_activation: bool = True):
        super().__init__()
        assert scale_factor in (2, 4, 8), "scale_factor debe ser 2, 4 u 8 (potencia de 2)"

        # Capa inicial
        self.initial = nn.Sequential(
            nn.KAN_Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # Bloques residuales (mismos hyperparams de norma/activación que indiques)
        blocks = [ResidualBlock(64, norm=norm, post_activation=post_activation)
                  for _ in range(num_residual_blocks)]
        self.res_blocks = nn.Sequential(*blocks)

        # Post-res (usa la misma norma que los residuales para consistencia)
        def _norm_layer(kind: str, c: int):
            if kind == 'bn':
                return nn.BatchNorm2d(c, affine=True)
            elif kind == 'in':
                return nn.InstanceNorm2d(c, affine=True, track_running_stats=False)
            elif kind == 'none':
                return nn.Identity()
            else:
                raise ValueError(f"norm inválido: {kind}")

        self.post_res = nn.Sequential(
            nn.KAN_Conv2d(64, 64, kernel_size=3, padding=1),
            _norm_layer(norm, 64)
        )

        # Upsample dinámico (una etapa por cada ×2)
        up_layers = []
        up_stages = int(math.log2(scale_factor))
        for _ in range(up_stages):
            # 64 -> 256 (=64*2*2) para PixelShuffle(2) y luego vuelve a 64 canales
            up_layers += [
                nn.KAN_Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        self.upsample = nn.Sequential(*up_layers)

        # Capa final
        self.final = nn.Sequential(
            nn.KAN_Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )

        # Inicialización
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.KAN_Conv2d):
                # No tocar: KAN_Conv2d no expone weight/bias como Conv2d
                continue
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.constant_(m.weight, 1.0)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.PReLU):
                if getattr(m, 'weight', None) is not None:
                    nn.init.constant_(m.weight, 0.25)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        initial = self.initial(x)

        # checkpoint en residuales — segmentos seguros
        n_blocks = len(self.res_blocks)
        segments = max(1, min(4, n_blocks))
        x = checkpoint_sequential(self.res_blocks, segments=segments, input=initial, use_reentrant=False)

        x = self.post_res(x) + initial
        x = self.upsample(x)
        return self.final(x)
