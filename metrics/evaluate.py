import math
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from metrics.lpips_metric import LPIPSMetric


def _psnr(x, y, eps=1e-12):
    """
    PSNR para imágenes en [0,1]
    """
    mse = F.mse_loss(x, y, reduction="mean").item()
    mse = max(mse, eps)
    return 10.0 * math.log10(1.0 / mse)


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """
    Normaliza automáticamente:
    - [-1,1] → [0,1]
    - [0,1] → [0,1]
    """
    if x.min() < 0:
        return ((x.clamp(-1, 1) + 1) * 0.5).clamp(0, 1)
    return x.clamp(0, 1)


def _to_y(x):
    """
    Convierte RGB → canal Y (luminancia)
    """
    if x.size(1) == 1:
        return x
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


@torch.no_grad()
def evaluate(model, loader, device, baseline_psnr=None, baseline_mssim=None):
    model.eval()

    lpips_metric = LPIPSMetric(device)

    total = {
        "psnr_rgb": 0.0,
        "psnr_y": 0.0,
        "mssim_rgb": 0.0,
        "mssim_y": 0.0,
        "lpips": 0.0
    }

    count = 0

    for lr, hr in loader:
        lr = lr.to(device)
        hr = hr.to(device)

        sr = model(lr)

        # Normalización robusta
        sr = _to_01(sr)
        hr = _to_01(hr)

        # Asegurar tamaños iguales
        if sr.shape[-2:] != hr.shape[-2:]:
            sr = F.interpolate(sr, size=hr.shape[-2:], mode="bilinear", align_corners=False)

        # -------- RGB --------
        psnr_rgb = _psnr(sr, hr)
        mssim_rgb = ms_ssim(sr, hr, data_range=1.0, size_average=True).item()

        # -------- Y --------
        sr_y = _to_y(sr)
        hr_y = _to_y(hr)

        psnr_y = _psnr(sr_y, hr_y)
        mssim_y = ms_ssim(sr_y, hr_y, data_range=1.0, size_average=True).item()

        # -------- LPIPS --------
        sr_lp = sr * 2 - 1
        hr_lp = hr * 2 - 1
        lp = lpips_metric(sr_lp, hr_lp)

        # Acumulación
        total["psnr_rgb"] += psnr_rgb
        total["psnr_y"] += psnr_y
        total["mssim_rgb"] += mssim_rgb
        total["mssim_y"] += mssim_y
        total["lpips"] += lp

        count += 1

    results = {k: v / count for k, v in total.items()}

    # Comparación vs baseline (muy útil para tu paper)
    if baseline_psnr is not None:
        results["delta_psnr_vs_pretrain"] = results["psnr_y"] - float(baseline_psnr)

    if baseline_mssim is not None:
        results["delta_mssim_vs_pretrain"] = results["mssim_y"] - float(baseline_mssim)

    return results