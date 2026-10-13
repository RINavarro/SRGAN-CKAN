import torch
import lpips

class LPIPSMetric:
    def __init__(self, device="cuda"):
        self.metric = lpips.LPIPS(net='alex').to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, sr, hr):
        # Asegurar rango [0,1]
        sr = sr.clamp(0, 1)
        hr = hr.clamp(0, 1)

        # Convertir a [-1,1]
        sr = sr * 2 - 1
        hr = hr * 2 - 1

        return self.metric(sr, hr).mean().item()