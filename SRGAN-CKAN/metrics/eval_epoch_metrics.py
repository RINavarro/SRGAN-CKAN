# %% ================== MÉTRICAS (PSNR-Y, MS-SSIM-Y, LPIPS opcional) ==================
import torch


def _to01(x: torch.Tensor) -> torch.Tensor:
    # robusto a [-1,1] o [0,1]
    if x.min() < 0 or x.max() > 1:
        return ((x.clamp(-1,1) + 1) * 0.5).clamp(0,1)
    return x.clamp(0,1)

def rgb_to_y(img):  # img: [B,3,H,W] en [0,1]
    r,g,b = img[:,0:1], img[:,1:2], img[:,2:3]
    y = 0.257*r + 0.504*g + 0.098*b + 16.0/255.0   # BT.601 rango [16/255,235/255]
    return y.clamp(0,1)

def center_crop(img, shave=4):
    if shave <= 0: return img
    return img[..., shave:-shave, shave:-shave]

@torch.no_grad()
def psnr_y(sr, hr, shave=4):
    sr01 = _to01(sr); hr01 = _to01(hr)
    sr_y = rgb_to_y(center_crop(sr01, shave))
    hr_y = rgb_to_y(center_crop(hr01, shave))
    mse = (sr_y - hr_y).pow(2).mean().clamp_min(1e-12)
    return 10.0*torch.log10(1.0/mse)

# MS-SSIM (si está torchmetrics instalado)
try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
    _ms_ssim = MS_SSIM(data_range=1.0).to('cuda' if torch.cuda.is_available() else 'cpu')
    @torch.no_grad()
    def ms_ssim_y(sr, hr, shave=4):
        sr01 = _to01(sr); hr01 = _to01(hr)
        sr_y = rgb_to_y(center_crop(sr01, shave))
        hr_y = rgb_to_y(center_crop(hr01, shave))
        return _ms_ssim(sr_y, hr_y).detach().cpu()
except Exception:
    @torch.no_grad()
    def ms_ssim_y(sr, hr, shave=4):
        return torch.tensor(float('nan'))

# LPIPS (opcional, pip install lpips)
try:
    import lpips
    _lpips_vgg = lpips.LPIPS(net='vgg').to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    @torch.no_grad()
    def lpips_score(sr, hr, shave=4):
        sr01 = _to01(sr); hr01 = _to01(hr)
        sr_c = center_crop(sr01, shave); hr_c = center_crop(hr01, shave)
        sr_n1 = (sr_c*2-1).to(torch.float32); hr_n1 = (hr_c*2-1).to(torch.float32)
        return _lpips_vgg(sr_n1, hr_n1).mean().cpu()
except Exception:
    @torch.no_grad()
    def lpips_score(sr, hr, shave=4):
        return torch.tensor(float('nan'))

# ---- Evaluación rápida por época en val_loader ----
def eval_epoch_metrics(generator, val_loader, device, shave=4, use_lpips=False):
    """
    Calcula PSNR-Y y MS-SSIM-Y (y LPIPS opcional) en val_loader.
    Devuelve dict con promedios.
    """
    from math import isnan
    generator.eval()
    psnr_list, ms_list, lp_list = [], [], []
    for lr, hr in val_loader:
        lr = lr.to(device); hr = hr.to(device)
        # usa tu helper autocast_safe si existe, si no FP32
        acm = globals().get("autocast_safe", None)
        ctx = acm(dtype=torch.bfloat16) if acm else torch.cuda.amp.autocast()
        with ctx:
            sr = generator(lr)
        psnr_list.append(psnr_y(sr, hr, shave=shave).item())
        ms_list.append(float(ms_ssim_y(sr, hr, shave=shave)))
        if use_lpips:
            lp_list.append(float(lpips_score(sr, hr, shave=shave)))
    generator.train()
    out = {
        "psnr_y": float(torch.tensor(psnr_list).mean()) if psnr_list else float('nan'),
        "ms_ssim_y": float(torch.tensor(ms_list).mean()) if ms_list else float('nan'),
    }
    if use_lpips:
        out["lpips"] = float(torch.tensor(lp_list).mean()) if lp_list else float('nan')
    return out