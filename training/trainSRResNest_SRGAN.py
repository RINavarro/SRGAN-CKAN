# %% =================== ENTRENAMIENTO CON PROGRESO (SRResNet / SRGAN) ===================
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from tqdm import tqdm

# ---------- Helper: autocast seguro (API nueva/vieja, BF16/FP16/FP32) ----------
def _gpu_supports_bf16() -> bool:
    """Devuelve True si la GPU soporta BF16 (Ampere+)."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 0)

def autocast_safe(dtype=torch.bfloat16, enabled=True):
    """
    Context manager de autocast a prueba de versiones:
    - Usa torch.amp.autocast('cuda', dtype=...) si existe (API nueva).
    - Si no, usa torch.cuda.amp.autocast(dtype=...) (API antigua).
    - Si no hay CUDA o enabled=False → nullcontext() (FP32).
    - Si la GPU no soporta BF16 → cae a FP16 (o puedes cambiar a FP32 retornando nullcontext()).
    """
    if not torch.cuda.is_available() or not enabled:
        return nullcontext()
    use_dtype = dtype
    if dtype == torch.bfloat16 and not _gpu_supports_bf16():
        use_dtype = torch.float16  # fallback razonable; si prefieres FP32, retorna nullcontext() aquí
    # API nueva
    try:
        from torch import amp
        return amp.autocast('cuda', dtype=use_dtype)
    except Exception:
        pass
    # API vieja
    try:
        from torch.cuda.amp import autocast
        return autocast(dtype=use_dtype)  # ojo: aquí NO va 'device_type'
    except Exception:
        return nullcontext()

# ---------- Formateador numérico para imprimir en la barra ----------
def _fmt(v):
    return f"{v:.4f}"

# ---------- Aliases externos que apunten a las versiones con progreso ----------
def train_srresnet(*args, **kwargs):
    """Alias público → llama a la versión con barra de progreso."""
    return _train_srresnet_progress(*args, **kwargs)

def train_srgan(*args, **kwargs):
    """Alias público → llama a la versión con barra de progreso."""
    return _train_srgan_progress(*args, **kwargs)

# ---------- SRResNet (pre-entrenamiento del generador) ----------
def _train_srresnet_progress(
    generator: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 20,
    accum_steps: int = 1,
    log_interval: int = 50
):
    # Crea scaler para AMP y define la pérdida de píxel (MSE)
    scaler = GradScaler()
    criterion = nn.MSELoss()
    history = {"loss": [], "psnr": []}

    # Modo entrenamiento + ajustes de rendimiento
    generator.train()
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except: pass
    generator.to(memory_format=torch.channels_last)  # layout amigable a Tensor Cores

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_psnr = 0.0
        nb = 0
        optimizer.zero_grad(set_to_none=True)

        # tqdm muestra una barra por época con métricas a la derecha
        pbar = tqdm(dataloader, desc=f"SRResNet | Epoch {epoch}/{epochs}", leave=True, dynamic_ncols=True)

        for step, (lr_imgs, hr_imgs) in enumerate(pbar, 1):
            # Sube tensores al device y usa channels_last para ganar rendimiento
            lr_imgs = lr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            hr_imgs = hr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)

            # Forward en mixed precision (si procede)
            with autocast_safe(dtype=torch.bfloat16):
                sr = generator(lr_imgs)                 # predicción SR
                loss = criterion(sr, hr_imgs) / accum_steps  # divide para gradient accumulation

            # Backward escalado
            scaler.scale(loss).backward()

            # Step cada 'accum_steps' mini-batches
            if step % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # ----- Métricas (PSNR) sin afectar al grafo -----
            with torch.no_grad():
                # Convertimos a [0,1] (equivalente a tu to_01) para medir PSNR
                sr01 = ((sr.detach().clamp(-1, 1) + 1) * 0.5).float().cpu()
                hr01 = ((hr_imgs.detach().clamp(-1, 1) + 1) * 0.5).float().cpu()
                mse = torch.mean((sr01 - hr01) ** 2, dim=(1, 2, 3)).clamp_min(1e-12).mean().item()
                psnr = 10.0 * math.log10(1.0 / mse)

            running_loss += loss.item() * accum_steps   # deshacer la división de accum
            running_psnr += psnr
            nb += 1

            # Actualiza barra cada cierto número de pasos
            if (step % log_interval == 0) or (step == len(dataloader)):
                pbar.set_postfix(loss=_fmt(running_loss / nb), psnr=_fmt(running_psnr / nb))

            # Libera activaciones del paso actual
            del sr

        # Guarda promedios por época
        epoch_loss = running_loss / max(1, nb)
        epoch_psnr = running_psnr / max(1, nb)
        history["loss"].append(epoch_loss)
        history["psnr"].append(epoch_psnr)
        print(f"✅ Epoch {epoch}: loss={_fmt(epoch_loss)} | PSNR={_fmt(epoch_psnr)} dB")
        torch.cuda.empty_cache()  # opcional: intenta liberar VRAM cacheada

    return history

# ---------- SRGAN (entrenamiento adversarial completo) ----------
def _train_srgan_progress(
    generator: nn.Module,
    discriminator: nn.Module,
    vgg: nn.Module,                      # tu VGGFeatureExtractor (eval/congelado por fuera)
    dataloader,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 20,
    accum_steps: int = 1,
    log_interval: int = 50,
    adv_w: float = 1e-3,                 # peso de pérdida adversarial (si cae PSNR, prueba 1e-4)
    cont_w: float = 1.0,                 # peso de pérdida de contenido (VGG)
    pix_w: float = 0.0                   # peso de L1 pixel (prueba 0.01 si necesitas estabilizar)
):
    # Modos correctos: G y D en train; VGG en eval (pero sin cortar grad en sr)
    generator.train(); discriminator.train(); vgg.eval()

    # Scalers AMP separados para G y D
    scaler_g = GradScaler(); scaler_d = GradScaler()

    # BCE sobre logits (¡NO uses Sigmoid en el Discriminador!)
    bce = nn.BCEWithLogitsLoss()

    # Registro de métricas por época
    history = {"gen_loss": [], "disc_loss": [], "adv_loss": [], "content_loss": [], "psnr": []}

    # Ajustes de rendimiento y layout de memoria
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except: pass
    generator.to(memory_format=torch.channels_last)
    discriminator.to(memory_format=torch.channels_last)

    for epoch in range(1, epochs + 1):
        g_run = d_run = adv_run = cont_run = psnr_run = 0.0
        nb = 0
        g_optimizer.zero_grad(set_to_none=True)
        d_optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(dataloader, desc=f"SRGAN     | Epoch {epoch}/{epochs}", leave=True, dynamic_ncols=True)
        for step, (lr_imgs, hr_imgs) in enumerate(pbar, 1):
            # Sube tensores a device y usa channels_last
            lr_imgs = lr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            hr_imgs = hr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)

            # ------------------- (1) Paso del Discriminador -------------------
            with autocast_safe(dtype=torch.bfloat16):
                sr_det = generator(lr_imgs).detach()            # SR sin gradiente para D (evita doble backward)
                real_logits = discriminator(hr_imgs)            # logits reales
                fake_logits = discriminator(sr_det)             # logits falsos
                d_loss = (bce(real_logits, torch.ones_like(real_logits)) +
                          bce(fake_logits, torch.zeros_like(fake_logits))) / accum_steps

            scaler_d.scale(d_loss).backward()                   # backprop de D con AMP
            if step % accum_steps == 0:
                scaler_d.step(d_optimizer)                      # step del optimizador D
                scaler_d.update()
                d_optimizer.zero_grad(set_to_none=True)         # limpia gradientes de D

            # ------------------- (2) Paso del Generador -------------------
            with autocast_safe(dtype=torch.bfloat16):
                sr = generator(lr_imgs)                         # SR con gradiente (actualiza G)
                fake_logits = discriminator(sr)                 # logits para SR
                adv_loss = bce(fake_logits, torch.ones_like(fake_logits))  # G quiere "1" (real)

            # VGG en FP32/estable se maneja dentro del extractor:
            vgg_sr = vgg(sr)                                    # ¡NO no_grad!: permite grad hacia sr
            with torch.no_grad():                               # solo ancla de contenido
                vgg_hr = vgg(hr_imgs)
            content_loss = F.mse_loss(vgg_sr, vgg_hr)           # pérdida perceptual (contenido VGG)
            pixel_loss   = F.l1_loss(sr, hr_imgs) if pix_w > 0 else sr.new_tensor(0.0)  # opcional

            # Pérdida total del generador ponderada
            g_loss = (adv_w * adv_loss + cont_w * content_loss + pix_w * pixel_loss) / accum_steps

            scaler_g.scale(g_loss).backward()                   # backprop de G con AMP
            if step % accum_steps == 0:
                scaler_g.step(g_optimizer)                      # step del optimizador G
                scaler_g.update()
                g_optimizer.zero_grad(set_to_none=True)         # limpia gradientes de G

            # ------------------- (3) Métricas y barra de progreso -------------------
            # PSNR sobre el batch actual (aprox; lo ideal es medir en val_loader aparte)
            with torch.no_grad():
                sr01 = ((sr.detach().clamp(-1, 1) + 1) * 0.5).float().cpu()
                hr01 = ((hr_imgs.detach().clamp(-1, 1) + 1) * 0.5).float().cpu()
                mse = torch.mean((sr01 - hr01) ** 2, dim=(1, 2, 3)).clamp_min(1e-12).mean().item()
                psnr = 10.0 * math.log10(1.0 / mse)

            # Acumula para promedio por época
            g_run   += float(g_loss.item() * accum_steps)
            d_run   += float(d_loss.item() * accum_steps)
            adv_run += float(adv_loss.item())
            cont_run+= float(content_loss.item())
            psnr_run+= float(psnr)
            nb += 1

            # Actualiza barra cada cierto número de pasos
            if (step % log_interval == 0) or (step == len(dataloader)):
                pbar.set_postfix(
                    D=_fmt(d_run / nb), G=_fmt(g_run / nb),
                    adv=_fmt(adv_run / nb), cont=_fmt(cont_run / nb),
                    psnr=_fmt(psnr_run / nb)
                )

            # Libera intermedios para mantener VRAM a raya
            del sr_det, sr, vgg_sr, vgg_hr, real_logits, fake_logits

        # ------------------- (4) Resumen por época -------------------
        eD = d_run / max(1, nb)
        eG = g_run / max(1, nb)
        eA = adv_run / max(1, nb)
        eC = cont_run / max(1, nb)
        eP = psnr_run / max(1, nb)
        history["disc_loss"].append(eD)
        history["gen_loss"].append(eG)
        history["adv_loss"].append(eA)
        history["content_loss"].append(eC)
        history["psnr"].append(eP)

        print(f"✅ Epoch {epoch}: D={_fmt(eD)} | G={_fmt(eG)} | adv={_fmt(eA)} | cont={_fmt(eC)} | PSNR={_fmt(eP)} dB")
        torch.cuda.empty_cache()  # opcional

    return history