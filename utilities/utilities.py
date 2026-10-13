# ------------------------- Autocast compatible (API nueva/vieja) -------------------------
import os

import torch
from torch import nn
from torchvision.utils import save_image


def _gpu_supports_bf16() -> bool:
    """Devuelve True si la GPU soporta BF16 de forma fiable (Ampere+)."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 0)               # Ampere (8.0+) o superior

def autocast_safe(dtype=torch.bfloat16, enabled=True):
    """
    Context manager de autocast que:
      - Usa torch.amp.autocast('cuda', ...) si está disponible (API nueva).
      - Si no, usa torch.cuda.amp.autocast(dtype=...) (API antigua).
      - Si no hay CUDA o enabled=False → nullcontext() (FP32).
      - Si la GPU no soporta BF16 → cae a FP16 (o FP32 si prefieres).
    """
    if not torch.cuda.is_available() or not enabled:
        return nullcontext()                      # no CUDA o deshabilitado → sin autocast (FP32)

    # Si se pide BF16 pero no hay soporte, cae a FP16 para mantener velocidad con cuidado
    use_dtype = dtype
    if dtype == torch.bfloat16 and not _gpu_supports_bf16():
        use_dtype = torch.float16                 # cambia a FP16; si no lo quieres, retorna nullcontext()

    # Intento API nueva
    try:
        from torch import amp                     # import local para evitar hard-dependencia
        return amp.autocast('cuda', dtype=use_dtype)
    except Exception:
        pass

    # Intento API antigua
    try:
        from torch.cuda.amp import autocast       # sin device_type en API antigua
        return autocast(dtype=use_dtype)
    except Exception:
        return nullcontext()                      # último recurso: FP32

# ------------------------- Utilidades de rango/PSNR -------------------------
def to_01(x: torch.Tensor) -> torch.Tensor:
    """
    Convierte tensores a rango [0,1] de forma robusta.
    Acepta entradas en [-1,1], [0,1] o rangos arbitrarios (normaliza por min/max por batch).
    """
    x_det = x.detach()
    xmin = float(x_det.min()); xmax = float(x_det.max())
    if xmin >= -1.01 and xmax <= 1.01 and (xmin < 0.0 or xmax > 1.0):
        x01 = (x_det + 1.0) * 0.5                 # de [-1,1] a [0,1]
    elif xmin >= 0.0 and xmax <= 1.0:
        x01 = x_det                               # ya en [0,1]
    else:
        x_min = x_det.amin(dim=(1,2,3), keepdim=True)
        x_max = x_det.amax(dim=(1,2,3), keepdim=True)
        x01 = (x_det - x_min) / (x_max - x_min + 1e-8)  # normaliza por batch
    return x01.clamp_(0, 1)                       # asegura límites exactos

def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Alias para código legacy: convención de 'denormalize' = mapear a [0,1]."""
    return to_01(x)

@torch.no_grad()
def psnr_batch(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Calcula PSNR promedio por batch en rango [0,1] (data_range=1.0).
    Desacopla a CPU/FP32 para estabilidad (no afecta gradientes).
    """
    sr01 = to_01(sr).to(dtype=torch.float32).cpu()
    hr01 = to_01(hr).to(dtype=torch.float32).cpu()
    mse = torch.mean((sr01 - hr01) ** 2, dim=(1,2,3)).clamp_min(1e-12)
    return (10.0 * torch.log10((data_range ** 2) / mse)).mean()

# ------------------------- Guardar ejemplos (LR | SR | HR) -------------------------
@torch.no_grad()
def save_examples(generator: nn.Module, dataloader, device, epoch_label: str = "final", out_dir: str = "results"):
    """
    Toma el primer batch del dataloader y guarda una rejilla con LR|SR|HR.
    Usa autocast seguro para acelerar la inferencia en GPU.
    """
    generator.eval()                              # modo evaluación para batchnorm/dropout, etc.
    os.makedirs(out_dir, exist_ok=True)           # crea carpeta si no existe
    try:
        lr_imgs, hr_imgs = next(iter(dataloader)) # saca un batch
    except StopIteration:
        return                                    # dataloader vacío → sal
    lr_imgs = lr_imgs.to(device, non_blocking=True)
    hr_imgs = hr_imgs.to(device, non_blocking=True)
    with autocast_safe(dtype=torch.bfloat16):     # inferencia mixta si es posible
        sr_imgs = generator(lr_imgs)              # genera super-resolución
    # Prepara rejilla concatenando LR, SR y HR (todos en [0,1])
    grid = torch.cat([denormalize(lr_imgs), denormalize(sr_imgs), denormalize(hr_imgs)], dim=0).cpu()
    save_image(grid, os.path.join(out_dir, f"samples_{epoch_label}.png"),
               nrow=lr_imgs.size(0), padding=2)   # guarda imagen (nrow = tamaño de batch)
    generator.train()                              # devolver a modo train