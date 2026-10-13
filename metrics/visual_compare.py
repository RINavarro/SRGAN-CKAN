import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _to_01(x: torch.Tensor) -> torch.Tensor:
    if x.min() < 0:
        return ((x.clamp(-1, 1) + 1) * 0.5).clamp(0, 1)
    return x.clamp(0, 1)


def _tensor_to_img(x):
    """
    Convierte tensor [C,H,W] → numpy [H,W,C]
    """
    return x.permute(1, 2, 0).numpy()


@torch.no_grad()
def save_comparisons(model_a, model_b, loader, device, out_dir, max_images=8):
    """
    Guarda comparaciones con títulos:
    LR | SRResNet | SRGAN-CKAN | HR
    """

    os.makedirs(out_dir, exist_ok=True)

    model_a.eval()
    model_b.eval()

    saved = 0

    for lr, hr in loader:
        lr = lr.to(device)
        hr = hr.to(device)

        sr_a = model_a(lr)
        sr_b = model_b(lr)

        lr_vis = _to_01(lr.detach().cpu())
        sr_a_vis = _to_01(sr_a.detach().cpu())
        sr_b_vis = _to_01(sr_b.detach().cpu())
        hr_vis = _to_01(hr.detach().cpu())

        for i in range(lr.size(0)):

            # 🔥 Upsample LR
            lr_up = F.interpolate(
                lr_vis[i].unsqueeze(0),
                size=hr_vis[i].shape[-2:],
                mode="nearest"
            ).squeeze(0)

            # Convertir a imágenes
            imgs = [
                _tensor_to_img(lr_up),
                _tensor_to_img(sr_a_vis[i]),
                _tensor_to_img(sr_b_vis[i]),
                _tensor_to_img(hr_vis[i]),
            ]

            titles = ["LR", "SRResNet", "SRGAN-CKAN", "HR"]

            # 🔥 Crear figura tipo paper
            fig, axes = plt.subplots(1, 4, figsize=(12, 4))

            for ax, img, title in zip(axes, imgs, titles):
                ax.imshow(img)
                ax.set_title(title, fontsize=12)
                ax.axis("off")

            # 🔥 Fondo blanco garantizado
            fig.patch.set_facecolor("white")

            # 🔥 Guardado de alta calidad
            out_path = os.path.join(out_dir, f"cmp_{saved:03d}.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")

            plt.close(fig)

            saved += 1

            if saved >= max_images:
                return