import torch
import torch.nn.functional as F


def extract_patch(img, x, y, w, h):
    """
    Extrae un parche de un tensor [B, C, H, W].
    """
    return img[..., y:y + h, x:x + w]


def resize_patch(patch, size):
    """
    Redimensiona un parche a un tamaño común.
    """
    return F.interpolate(
        patch,
        size=size,
        mode="bilinear",
        align_corners=False,
    )


def create_zoom_row(lr, sr, hr, region, scale=4, zoom_factor=4):
    """
    Crea una fila:
    LR zoom | SR zoom | HR zoom

    La región se define en coordenadas HR/SR.
    """
    x, y, w, h = region

    # Coordenadas equivalentes en LR
    lr_x = x // scale
    lr_y = y // scale
    lr_w = max(1, w // scale)
    lr_h = max(1, h // scale)

    lr_patch = extract_patch(lr, lr_x, lr_y, lr_w, lr_h)
    sr_patch = extract_patch(sr, x, y, w, h)
    hr_patch = extract_patch(hr, x, y, w, h)

    out_size = (h * zoom_factor, w * zoom_factor)

    lr_zoom = resize_patch(lr_patch, out_size)
    sr_zoom = resize_patch(sr_patch, out_size)
    hr_zoom = resize_patch(hr_patch, out_size)

    return torch.cat([lr_zoom, sr_zoom, hr_zoom], dim=-1)


def create_zoom_grid(lr, sr, hr, regions, scale=4, zoom_factor=4):
    """
    Crea una grilla vertical con varias regiones de zoom.
    """
    rows = [
        create_zoom_row(
            lr,
            sr,
            hr,
            region,
            scale=scale,
            zoom_factor=zoom_factor,
        )
        for region in regions
    ]

    return torch.cat(rows, dim=-2)