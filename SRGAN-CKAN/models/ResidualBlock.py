from torch import nn


class ResidualBlock(nn.Module):
    """
    Bloque residual para SR:
    - norm: 'in' (InstanceNorm, recomendado con batch pequeño), 'bn' (BatchNorm) o 'none'
    - post_activation: si True, aplica PReLU después de la suma residual
    """

    def __init__(self, channels, norm: str = 'in', post_activation: bool = True):
        super().__init__()

        def _norm_layer(kind: str, c: int):
            if kind == 'bn':
                return nn.BatchNorm2d(c, affine=True)
            elif kind == 'in':
                # IN estable para batch pequeño; sin running stats
                return nn.InstanceNorm2d(c, affine=True, track_running_stats=False)
            elif kind == 'none':
                return nn.Identity()
            else:
                raise ValueError(f"norm inválido: {kind}")

        N = lambda: _norm_layer(norm, channels)

        self.block = nn.Sequential(
            nn.KAN_Conv2d(channels, channels, kernel_size=3, padding=1),
            N(),
            nn.PReLU(),
            nn.KAN_Conv2d(channels, channels, kernel_size=3, padding=1),
            N()
        )

        # Activación después de la suma
        self.post_act = nn.PReLU() if post_activation else nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.KAN_Conv2d):
                # Importante: no inicializar como si fuera Conv2d estándar
                continue
            elif isinstance(m, nn.Conv2d):
                # Kaiming para conv + (norm) + PReLU: fan_in es razonable
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                # Asegura identidad al inicio
                if getattr(m, 'weight', None) is not None:
                    nn.init.constant_(m.weight, 1.0)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.PReLU):
                # Slope inicial típica
                if getattr(m, 'weight', None) is not None:
                    nn.init.constant_(m.weight, 0.25)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = x + self.block(x)
        return self.post_act(out)
