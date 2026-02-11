import torch
from torch import nn
import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
from tqdm import tqdm

from training.trainSRResNest_SRGAN import autocast_safe

def psnr_batch(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    PSNR promedio por batch sobre tensores [N, C, H, W].
    Asume rango [0, max_val]; se hace clamp para evitar valores fuera de rango.
    """
    sr_clamped = sr.clamp(0.0, max_val)
    hr_clamped = hr.clamp(0.0, max_val)
    mse = torch.mean((sr_clamped - hr_clamped) ** 2, dim=[1, 2, 3])  # [N]
    psnr = 10.0 * torch.log10((max_val ** 2) / (mse + 1e-8))
    return psnr.mean()  # escalar

def train_srresnet_amp(generator, dataloader, optimizer, device, epochs=20, accum_steps: int = 1):
    """
    Pre-entrena el Generador con pérdida MSE (SRResNet).
    - Mixed precision segura via autocast_safe
    - Accumulation steps para ahorrar memoria si se requiere
    - PSNR por época (promedio de entrenamiento; lo ideal es medir validación aparte)
    """
    generator.train()                              # modo entrenamiento
    criterion = nn.MSELoss()                       # pérdida de píxel (MSE clásico de SRResNet)
    scaler = GradScaler()                          # escalado de gradiente para AMP
    history = {"loss": [], "psnr": []}             # registro básico de métricas

    # Optimizaciones de backend (no fallan si no están disponibles)
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    # Sugiere channels_last para convoluciones 2D en GPU (mejor memoria/cache)
    generator.to(device)
    generator.to(memory_format=torch.channels_last)

    for epoch in range(epochs):                    # bucle de épocas
        run_loss = 0.0; run_psnr = 0.0; nb = 0     # acumuladores para promedios
        optimizer.zero_grad(set_to_none=True)      # limpia gradientes al inicio

        for step, (lr_imgs, hr_imgs) in enumerate(dataloader):    # itera batches
            # Mover a GPU (si hay), activar pinned memory (si pin_memory=True en DataLoader)
            lr_imgs = lr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            hr_imgs = hr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)

            # Forward en mixed precision (si está habilitado) para acelerar y ahorrar VRAM
            with autocast_safe(dtype=torch.bfloat16):
                sr = generator(lr_imgs)                          # predicción SR
                loss = criterion(sr, hr_imgs) / accum_steps      # divide para acumulación de gradiente

            # Backward con GradScaler (evita underflow/overflow en FP16/BF16)
            scaler.scale(loss).backward()

            # Step cada 'accum_steps' minibatches
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)       # aplica paso con scaling
                scaler.update()              # actualiza factor de escala
                optimizer.zero_grad(set_to_none=True)  # limpia gradientes

            # Métricas (sin gradiente; detach implícito por psnr_batch)
            run_loss += loss.item() * accum_steps
            run_psnr += float(psnr_batch(sr, hr_imgs).item())
            nb += 1

            del sr                              # libera memoria intermedia

        # Al cierre de la época, guarda promedios
        history["loss"].append(run_loss / max(1, nb))
        history["psnr"].append(run_psnr / max(1, nb))
        torch.cuda.empty_cache()                # hint para liberar caché entre épocas (opcional)

    return history