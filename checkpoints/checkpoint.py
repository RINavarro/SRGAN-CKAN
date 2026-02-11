# %% ================== CHECKPOINTING ==================
import os

import torch


def save_checkpoint(epoch, phase, generator, discriminator, g_opt, d_opt, history, path="checkpoints/srgan.pt"):
    """
    Guarda estado completo para reanudar:
    - epoch: total de épocas completadas dentro de la fase actual
    - phase: 'pretrain' o 'gan' (o 'final')
    - modelos, optimizadores y 'history' acumulado (srresnet/srgan)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": int(epoch),
        "phase": str(phase),
        "generator": generator.state_dict(),
        "g_opt": g_opt.state_dict() if g_opt is not None else None,
        "history": history if history is not None else {},
    }
    if discriminator is not None:
        ckpt["discriminator"] = discriminator.state_dict()
    if d_opt is not None:
        ckpt["d_opt"] = d_opt.state_dict()
    torch.save(ckpt, path)

def load_checkpoint(path, generator, discriminator=None, g_opt=None, d_opt=None, map_location="auto"):
    """
    Carga checkpoint si existe. Devuelve (epoch, phase, history).
    - Si no hay archivo: (0, 'none', {}).
    """
    if not os.path.exists(path):
        return 0, "none", {}
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=map_location)
    generator.load_state_dict(ckpt["generator"])
    if g_opt is not None and ckpt.get("g_opt") is not None:
        g_opt.load_state_dict(ckpt["g_opt"])
    if discriminator is not None and ckpt.get("discriminator") is not None:
        discriminator.load_state_dict(ckpt["discriminator"])
    if d_opt is not None and ckpt.get("d_opt") is not None:
        d_opt.load_state_dict(ckpt["d_opt"])
    return int(ckpt.get("epoch", 0)), ckpt.get("phase", "none"), ckpt.get("history", {})