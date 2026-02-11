import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights


class VGGFeatureExtractor(nn.Module):
    """
    Extractor de características VGG19 para pérdida perceptual.
    - Usa API moderna: weights=VGG19_Weights.IMAGENET1K_V1
    - Extrae hasta la capa ~conv5_4 (features[:35])
    - Ejecución en FP32 (autocast deshabilitado) para estabilidad
    - SIN no_grad: permite gradiente hacia la entrada (el G aprende)
    - Normaliza la entrada a estadísticas ImageNet
    """

    def __init__(self):
        super().__init__()
        # Cargar VGG19 con pesos ImageNet
        base = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # Hasta ReLU22 / conv5_4 aprox (35 es habitual en implementaciones SRGAN)
        self.vgg = base[:35].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False  # no se entrenan los pesos de VGG

        # Buffers de normalización (se mueven al device automáticamente con .to())
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _to_01(self, x: torch.Tensor) -> torch.Tensor:
        # Convierte de [-1,1] o rango arbitrario a [0,1] de forma segura
        xmin = float(x.detach().min())
        xmax = float(x.detach().max())
        if xmin >= -1.01 and xmax <= 1.01 and (xmin < 0.0 or xmax > 1.0):
            y = (x + 1.0) * 0.5
        elif xmin >= 0.0 and xmax <= 1.0:
            y = x
        else:
            x_min = x.amin(dim=(1, 2, 3), keepdim=True)
            x_max = x.amax(dim=(1, 2, 3), keepdim=True)
            y = (x - x_min) / (x_max - x_min + 1e-8)
        return y.clamp(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna características en FP32.
        No usar torch.no_grad aquí: queremos backprop hacia la entrada (G).
        """
        # Forzar FP32 y desactivar autocast dentro de VGG para estabilidad
        with torch.autocast(device_type='cuda', enabled=False) if x.is_cuda else torch.autocast(enabled=False):
            x = self._to_01(x)
            # Asegurar dtype/disp de mean/std
            mean = self.mean.to(dtype=x.dtype, device=x.device)
            std = self.std.to(dtype=x.dtype, device=x.device)
            # Normalización ImageNet
            x = (x - mean) / (std + 1e-8)
            # VGG en eval/FP32
            x = x.to(dtype=torch.float32)
            feats = self.vgg(x)
        return feats  # fp32 (el loss se calcula estable y backprop fluye hacia la entrada)