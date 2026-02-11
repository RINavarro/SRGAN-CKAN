import os, math, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

# ---- Fallbacks suaves por si no pegaste los helpers previos en este notebook ----
def _to01_fallback(x: torch.Tensor) -> torch.Tensor:
    """Fallback simple a [0,1] si no existe tu denormalize/to_01."""
    x_det = x.detach()
    xmin = float(x_det.min()); xmax = float(x_det.max())
    if xmin >= -1.01 and xmax <= 1.01 and (xmin < 0.0 or xmax > 1.0):
        y = (x_det + 1.0) * 0.5
    elif xmin >= 0.0 and xmax <= 1.0:
        y = x_det
    else:
        x_min = x_det.amin(dim=(1,2,3), keepdim=True)
        x_max = x_det.amax(dim=(1,2,3), keepdim=True)
        y = (x_det - x_min) / (x_max - x_min + 1e-8)
    return y.clamp_(0,1)

# Si ya definiste denormalize / autocast_safe en celdas previas, se usarán.
denormalize = globals().get("denormalize", _to01_fallback)
autocast_safe = globals().get("autocast_safe", lambda **kw: torch.cuda.amp.autocast())

# ------------------------------------------------------------------------------------------
def plot_training_history(history: dict, tag: str = ""):
    """
    Grafica métricas por época de forma robusta:
    - Pérdidas (gen/disc)
    - Componentes (content/adversarial) si existen
    - Calidad (PSNR y SSIM) si existen
    Guarda 'results/history_{tag}.png' y muestra la figura.
    """
    # Extrae longitudes y maneja faltantes con valores vacíos
    epochs = range(1, len(history.get("gen_loss", [])) + 1)
    gen_loss = history.get("gen_loss", [])
    disc_loss = history.get("disc_loss", [])
    content_loss = history.get("content_loss", [])
    adv_loss = history.get("adv_loss", [])
    psnr = history.get("psnr", [])
    ssim = history.get("ssim", [])  # podría no existir

    # Estilo oscuro para mejor contraste
    plt.style.use("dark_background")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # === 1. Training Losses ===
    if gen_loss:
        axs[0].plot(epochs, gen_loss, label="Generator Loss", color="#00bfff")
    if disc_loss:
        axs[0].plot(epochs, disc_loss, label="Discriminator Loss", color="#ff7f50")
    axs[0].set_title("Training Losses")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # === 2. Component Losses ===
    if content_loss:
        axs[1].plot(epochs, content_loss, label="Content Loss", color="#87cefa")
    if adv_loss:
        axs[1].plot(epochs, adv_loss, label="Adversarial Loss", color="#ffa07a")
    axs[1].set_title("Component Losses")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].grid(True, linestyle="--", alpha=0.5)

    # === 3. Quality Metrics ===
    if psnr:
        axs[2].plot(epochs, psnr, label="PSNR (dB)", color="#00ff7f")
    if ssim:
        axs[2].plot(epochs, ssim, label="SSIM", color="#ffd700")
    axs[2].set_title("Quality Metrics")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Score")
    axs[2].legend()
    axs[2].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"SRGAN Training Summary - {tag}", fontsize=16)
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"results/history_{tag}.png", dpi=150)
    plt.show()

# ------------------------------------------------------------------------------------------
def visualize_triplet_grid(lr_imgs, sr_imgs, hr_imgs, save_path: str = None, title: str = None):
    """
    Dibuja una grilla con 3B imágenes:
      - Fila 1: LR (redimensionadas a tamaño SR/HR)
      - Fila 2: SR (salida del generador)
      - Fila 3: HR (ground truth)
    Si 'save_path' se provee, guarda la figura. Siempre muestra (plt.show()).
    """
    # Normaliza a [0,1] con tu denormalize (o fallback)
    lr_imgs = denormalize(lr_imgs)
    sr_imgs = denormalize(sr_imgs)
    hr_imgs = denormalize(hr_imgs)

    # Redimensiona LR para que coincida visualmente con SR/HR
    _, _, H, W = sr_imgs.shape
    lr_resized = F.interpolate(lr_imgs, size=(H, W), mode="bilinear", align_corners=False)

    # Concatena para tener 3B imágenes (LR|SR|HR) y arma la grilla por batch
    imgs = torch.cat([lr_resized, sr_imgs, hr_imgs], dim=0)  # [3B, C, H, W]
    grid = make_grid(imgs, nrow=lr_imgs.size(0), padding=2) # una fila por conjunto LR/SR/HR
    np_img = grid.permute(1, 2, 0).cpu().numpy()            # CHW → HWC para matplotlib

    # Render
    plt.figure(figsize=(15, 5))
    plt.imshow(np_img)
    plt.axis("off")
    if title:
        plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------------------------------------
@torch.no_grad()
def save_examples(generator: torch.nn.Module, dataloader, device, epoch_label: str = "final", max_batches: int = 1):
    """
    Genera y guarda comparativas LR|SR|HR de los primeros 'max_batches' del dataloader.
    - 'epoch_label' aparece en el nombre del archivo para rastrear el progreso.
    - Usa autocast_safe para acelerar la inferencia si hay GPU/AMP disponible.
    """
    generator.eval()                                  # eval para BatchNorm/Dropout en inferencia
    os.makedirs("results", exist_ok=True)

    for bidx, (lr_imgs, hr_imgs) in enumerate(dataloader):
        if bidx >= max_batches:                       # limita cuántos batches guardas por llamada
            break

        # Mueve a device una sola vez; usa autocast para la predicción SR
        lr_dev = lr_imgs.to(device, non_blocking=True)
        with autocast_safe(dtype=torch.bfloat16):
            sr_dev = generator(lr_dev)

        # Mueve todo a CPU sólo para visualización/guardado
        lr_cpu = lr_imgs.cpu()
        sr_cpu = sr_dev.cpu()
        hr_cpu = hr_imgs.cpu()

        # Componer título y path del archivo
        title = f"Epoch {epoch_label} - Batch {bidx} (LR → SR → HR)"
        save_path = f"results/comparison_{epoch_label}_batch{bidx}.png"

        # Visualiza/graba la grilla (usa denormalize y resize interno)
        visualize_triplet_grid(lr_cpu, sr_cpu, hr_cpu, save_path=save_path, title=title)

# ------------------------------------------------------------------------------------------
def save_examples_every(epoch: int,
                        generator: torch.nn.Module,
                        val_loader,                      # mejor usa el de validación para muestras “limpias”
                        device: torch.device,
                        every: int = 10,
                        max_batches: int = 1,
                        prefix: str = "srgan"):
    """
    Llama a 'save_examples' automáticamente cada 'every' épocas (p.ej. cada 10).
    """
    # Si no ha tocado el múltiplo configurado, no hace nada
    if every is None or every <= 0:
        return
    if (epoch % every) != 0:
        return

    # Genera etiqueta compacta para los archivos (p.ej., srgan_e10)
    epoch_label = f"{prefix}_e{epoch}"
    save_examples(generator, val_loader, device, epoch_label=epoch_label, max_batches=max_batches)

