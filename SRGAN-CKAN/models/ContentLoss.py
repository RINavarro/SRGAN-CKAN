import torch
from torch import nn


class ContentLoss(nn.Module):
    """
    Pérdida de contenido (perceptual) con VGG:
    - Usa el feature_extractor que ya congelaste y ejecutas en FP32.
    - Si sr y hr difieren en tamaño, hace resize bilineal sin gradientes raros.
    """
    def __init__(self, feature_extractor: nn.Module, reduction: str = "mean"):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.crit = nn.MSELoss(reduction=reduction)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        if sr.shape[2:] != hr.shape[2:]:
            # evita align_corners=True (inestable para SR)
            sr = F.interpolate(sr, size=hr.shape[2:], mode="bilinear", align_corners=False)
        # vgg() ya corre en FP32 dentro del extractor (sin autocast)
        feats_sr = self.feature_extractor(sr)          # grad fluye hacia sr (G aprende)
        feats_hr = self.feature_extractor(hr).detach() # ancla de contenido sin gradiente
        return self.crit(feats_sr, feats_hr)


class AdversarialLoss(nn.Module):
    """
    BCEWithLogits con label smoothing determinista.
    Evita ruido aleatorio por batch; más estable y reproducible.
    """
    def __init__(self, smooth_real: float = 0.1, smooth_fake: float = 0.0):
        """
        smooth_real: poner targets reales en 1 - smooth_real  (e.g. 0.9)
        smooth_fake: poner targets falsos en 0 + smooth_fake (e.g. 0.0 ~ 0.1)
        """
        super().__init__()
        self.smooth_real = float(smooth_real)
        self.smooth_fake = float(smooth_fake)
        self.bce = nn.BCEWithLogitsLoss()

    def targets(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        if is_real:
            val = 1.0 - self.smooth_real
        else:
            val = 0.0 + self.smooth_fake
        return torch.full_like(pred, fill_value=val)

    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        tgt = self.targets(pred, is_real)
        return self.bce(pred, tgt)