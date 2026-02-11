import torch
from torch import GradScaler

from training.trainSRResNest_SRGAN import autocast_safe
from training.train_srresnet_amp import train_srresnet_amp


def train_srgan_amp(generator, discriminator, vgg, dataloader, g_optimizer, d_optimizer,
                    device, epochs=20, accum_steps:int=1,
                    adv_w:float=1e-3, cont_w:float=1.0, pix_w:float=0.0):
    """
    Entrena SRGAN con pérdidas ponderadas:
      G: adv_w * BCEWithLogits(fake, 1) + cont_w * MSE(VGG(sr), VGG(hr)) + pix_w * L1(sr, hr)
      D: BCEWithLogits(real,1) + BCEWithLogits(fake,0)
    - Usar logits (Discriminator SIN Sigmoid)
    - VGG debe estar en eval y congelado; el gradiente fluye a través de sr (NO usar no_grad en VGG(sr))
    """
    generator.train(); discriminator.train(); vgg.eval()  # modos apropiados de cada módulo
    scaler_g = GradScaler(); scaler_d = GradScaler()      # escaladores para G y D
    bce = nn.BCEWithLogitsLoss()                          # BCE sobre logits (sin Sigmoid en D)
    history = {"gen_loss": [], "disc_loss": [], "adv_loss": [], "content_loss": [], "psnr": []}

    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    # Sugerir channels_last para operaciones 2D
    generator.to(memory_format=torch.channels_last)
    discriminator.to(memory_format=torch.channels_last)

    for epoch in range(epochs):
        g_run = d_run = adv_run = cont_run = psnr_run = 0.0  # acumuladores
        nb = 0
        g_optimizer.zero_grad(set_to_none=True)
        d_optimizer.zero_grad(set_to_none=True)

        for step, (lr_imgs, hr_imgs) in enumerate(dataloader):
            # Move a device y formato de memoria amigable con Tensor Cores
            lr_imgs = lr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            hr_imgs = hr_imgs.to(device, non_blocking=True, memory_format=torch.channels_last)

            # ---- (1) Actualización del Discriminador ----
            with autocast_safe(dtype=torch.bfloat16):
                sr_det = generator(lr_imgs).detach()          # SR sin gradiente para D (evita doble backward a G)
                real = discriminator(hr_imgs)                 # logits de reales
                fake = discriminator(sr_det)                  # logits de falsos
                d_loss = (bce(real, torch.ones_like(real))    # etiqueta real=1
                          + bce(fake, torch.zeros_like(fake)) # etiqueta falso=0
                         ) / accum_steps                      # división para acumulación

            scaler_d.scale(d_loss).backward()                 # backprop D (escalado)
            if (step + 1) % accum_steps == 0:
                scaler_d.step(d_optimizer)                    # aplica paso D
                scaler_d.update()
                d_optimizer.zero_grad(set_to_none=True)       # limpia gradientes de D

            # ---- (2) Actualización del Generador ----
            with autocast_safe(dtype=torch.bfloat16):
                sr = generator(lr_imgs)                       # SR con gradiente (G se actualiza)
                fake = discriminator(sr)                      # logits predichos por D para SR
                adv_loss = bce(fake, torch.ones_like(fake))   # G quiere que D diga "real" (1)

            # VGG en FP32 lo gestiona internamente tu extractor; grad **SÍ** debe fluir desde VGG(sr)
            vgg_sr = vgg(sr)                                  # NO usar no_grad: permite gradiente a sr
            with torch.no_grad():                             # para hr no necesitamos gradiente
                vgg_hr = vgg(hr_imgs)
            content_loss = F.mse_loss(vgg_sr, vgg_hr)         # pérdida perceptual (contenido)
            pixel_loss   = F.l1_loss(sr, hr_imgs) if pix_w > 0 else sr.new_tensor(0.0)  # opcional

            g_loss = (adv_w * adv_loss +                      # pondera adversarial
                      cont_w * content_loss +                 # pondera contenido (VGG)
                      pix_w  * pixel_loss) / accum_steps      # pondera píxel (si se activa)

            scaler_g.scale(g_loss).backward()                 # backprop G (escalado)
            if (step + 1) % accum_steps == 0:
                scaler_g.step(g_optimizer)                    # aplica paso G
                scaler_g.update()
                g_optimizer.zero_grad(set_to_none=True)       # limpia gradientes de G

            # ---- (3) Logging/métricas por batch ----
            g_run   += float(g_loss.item() * accum_steps)     # suma de pérdidas (des-escala)
            d_run   += float(d_loss.item() * accum_steps)
            adv_run += float(adv_loss.item())
            cont_run+= float(content_loss.item())
            psnr_run+= float(psnr_batch(sr, hr_imgs).item())
            nb += 1

            # Libera tensores intermedios para evitar picos de memoria
            del sr_det, sr, vgg_sr, vgg_hr, real, fake

        # ---- (4) Resumen por época ----
        history["gen_loss"].append(g_run / max(1, nb))
        history["disc_loss"].append(d_run / max(1, nb))
        history["adv_loss"].append(adv_run / max(1, nb))
        history["content_loss"].append(cont_run / max(1, nb))
        history["psnr"].append(psnr_run / max(1, nb))

        torch.cuda.empty_cache()                               # suelta caché (opcional)

    return history

# ------------------------- Wrappers con nombres legacy -------------------------
def train_srresnet(*args, **kwargs):
    """Alias para mantener firma original """
    return train_srresnet_amp(*args, **kwargs)

def train_srgan(*args, **kwargs):
    """Alias para mantener firma original """
    return train_srgan_amp(*args, **kwargs)