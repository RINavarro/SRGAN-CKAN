from torch import nn


class Discriminator(nn.Module):
    """
    Discriminador SRGAN con KAN_Conv2d:
    - Todos los conv son nn.KAN_Conv2d
    - Primer bloque sin normalización; siguientes con 'in' (InstanceNorm) por defecto
    - Sin Sigmoid al final (salida = logits). Usa BCEWithLogitsLoss en el loop.
    - Downsample con conv 3x3 stride=2 como en SRGAN original
    """

    def __init__(self, input_shape=(3, 256, 256), norm: str = "in"):
        super().__init__()
        channels, _, _ = input_shape

        def _norm(kind: str, c: int):
            if kind == "bn":
                return nn.BatchNorm2d(c, affine=True)
            elif kind == "in":
                # estable con batch pequeño
                return nn.InstanceNorm2d(c, affine=True, track_running_stats=False)
            elif kind == "none":
                return nn.Identity()
            else:
                raise ValueError(f"norm inválido: {kind}")

        def disc_block(in_f, out_f, first_block=False):
            layers = []
            # conv 3x3, stride=1
            layers.append(nn.KAN_Conv2d(in_f, out_f, kernel_size=3, stride=1, padding=1))
            if not first_block and norm != "none":
                layers.append(_norm(norm, out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            # conv 3x3, stride=2 (downsample)
            layers.append(nn.KAN_Conv2d(out_f, out_f, kernel_size=3, stride=2, padding=1))
            if norm != "none":
                layers.append(_norm(norm, out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 4 bloques como en SRGAN: 64,128,256,512
        feats = []
        in_f = channels
        for i, out_f in enumerate([64, 128, 256, 512]):
            feats.extend(disc_block(in_f, out_f, first_block=(i == 0)))
            in_f = out_f
        self.features = nn.Sequential(*feats)

        # Cabeza final (logits, sin Sigmoid)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1, bias=True)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            # No inicializar KAN_Conv2d como Conv2d
            if isinstance(m, nn.KAN_Conv2d):
                continue
            elif isinstance(m, nn.Conv2d):
                # Por si en algún lado usas Conv2d normal
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.constant_(m.weight, 1.0)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, img):
        x = self.features(img)
        logits = self.classifier(x)  # sin Sigmoid
        return logits