import gc
import os

import torch
from torch import optim, nn

from checkpoints.checkpoint import load_checkpoint, save_checkpoint
from metrics.eval_epoch_metrics import eval_epoch_metrics
from models.Discriminator import Discriminator
from models.Generator import Generator
from models.VGGFeatureExtractor import VGGFeatureExtractor
from training.training import plot_training_history, save_examples
from training.trainSRResNest_SRGAN import train_srresnet, train_srgan


def main():
    # ----------------- Dispositivo y preferencias GPU -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

    # --------------------- Configuración de ejecución ---------------------
    HR_SIZE = 256
    BATCH   = 4
    NUM_WORKERS = 2
    PREFETCH    = 2
    TIMEOUT     = 60

    PRE_EPOCHS  = 20
    GAN_EPOCHS  = 20
    SHOW_EVERY  = 20                  # muestras cada N épocas
    CKPT_PATH   = "checkpoints/srgan.pt"

    # ----------------------------- DataLoaders -----------------------------
    train_loader = DataLoader(
        SRGANDataset(split='train', hr_size=HR_SIZE, max_samples=1600),
        batch_size=BATCH, shuffle=True,
        num_workers=NUM_WORKERS, persistent_workers=True,
        prefetch_factor=PREFETCH, timeout=TIMEOUT, pin_memory=True
    )
    val_loader = DataLoader(
        SRGANDataset(split='validation', hr_size=HR_SIZE, max_samples=1600),
        batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, persistent_workers=True,
        prefetch_factor=PREFETCH, timeout=TIMEOUT, pin_memory=True
    )

    # ------------------------------- Modelos -------------------------------
    generator = Generator().to(device).to(memory_format=torch.channels_last)
    discriminator = Discriminator().to(device).to(memory_format=torch.channels_last)
    vgg = VGGFeatureExtractor().to(device).eval()

    # -------- (Opcional) Normaliza chunk_pixels de KAN_Conv2d --------------
    def _norm_chunk(kan_layer, H, W):
        if not hasattr(kan_layer, "chunk_pixels"): return
        kH,kW = kan_layer.kernel_size; sH,sW = kan_layer.stride
        pH,pW = kan_layer.padding;    dH,dW = kan_layer.dilation
        W_out = (W + 2*pW - dW*(kW - 1) - 1)//sW + 1
        per_row = max(1, W_out)
        max_rows = max(1, kan_layer.chunk_pixels // per_row)
        kan_layer.chunk_pixels = max_rows * per_row

    def _norm_all_kan(model, H, W):
        for m in model.modules():
            if isinstance(m, nn.KAN_Conv2d):
                _norm_chunk(m, H, W)

    _norm_all_kan(generator, HR_SIZE, HR_SIZE)
    _norm_all_kan(discriminator, HR_SIZE, HR_SIZE)

    # ------------------------------ Optimizadores ---------------------------
    g_optimizer = optim.Adam(generator.parameters(),     lr=1e-4, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # ----------------------------- Reanudación ------------------------------
    # intenta cargar checkpoint (si no existe, queda en (0,'none',{}))
    start_epoch, start_phase, ckpt_hist = load_checkpoint(
        CKPT_PATH, generator, discriminator, g_optimizer, d_optimizer
    )
    print(f"🔁 Reanudación: epoch={start_epoch}, phase='{start_phase}'")

    # Historiales acumulados
    srresnet_history = ckpt_hist.get("srresnet", {"loss": [], "psnr": []})
    srgan_history    = ckpt_hist.get("srgan",    {"gen_loss": [], "disc_loss": [], "adv_loss": [], "content_loss": [], "psnr": []})

    # ====================== PRE-ENTRENAMIENTO: SRResNet =====================
    # Si ya se completó (o la fase guardada es 'gan'), saltamos
    if start_phase in ("none", "pretrain") and len(srresnet_history["loss"]) < PRE_EPOCHS:
        print("🛠️ Pre-entrenamiento SRResNet...")
        done = len(srresnet_history["loss"])
        remaining = PRE_EPOCHS - done
        while remaining > 0:
            run = min(SHOW_EVERY, remaining)
            hist = train_srresnet(generator, train_loader, g_optimizer, device, epochs=run)
            srresnet_history["loss"].extend(hist.get("loss", []))
            srresnet_history["psnr"].extend(hist.get("psnr", []))
            done += run
            # muestras y métricas de validación por bloque
            save_examples(generator, val_loader, device, epoch_label=f"srresnet_e{done}", max_batches=1)
            valm = eval_epoch_metrics(generator, val_loader, device)
            print(f"🔎 Val (Y): PSNR={valm['psnr_y']:.2f} dB | MS-SSIM={valm['ms_ssim_y']:.4f}")

            # checkpoint al final del bloque de pretrain
            ckpt_hist["srresnet"] = srresnet_history
            save_checkpoint(done, "pretrain", generator, None, g_optimizer, None, ckpt_hist, path=CKPT_PATH)
            remaining -= run
        start_phase = "gan"  # avanzamos de fase

    # =================== ENTRENAMIENTO ADVERSARIAL: SRGAN ===================
    if start_phase in ("gan", "final") and len(srgan_history["gen_loss"]) < GAN_EPOCHS:
        print("\n⚔️ Entrenamiento SRGAN...")
        done = len(srgan_history["gen_loss"])
        remaining = GAN_EPOCHS - done
        while remaining > 0:
            run = min(SHOW_EVERY, remaining)
            hist = train_srgan(generator, discriminator, vgg, train_loader,
                               g_optimizer, d_optimizer, device, epochs=run)
            # acumula historia
            for k in srgan_history.keys():
                if k in hist: srgan_history[k].extend(hist[k])
            done += run

            # muestras y métricas por bloque
            save_examples(generator, val_loader, device, epoch_label=f"srgan_e{done}", max_batches=1)
            valm = eval_epoch_metrics(generator, val_loader, device)
            print(f"🔎 Val (Y): PSNR={valm['psnr_y']:.2f} dB | MS-SSIM={valm['ms_ssim_y']:.4f}")

            # checkpoint al final del bloque GAN
            ckpt_hist["srgan"] = srgan_history
            save_checkpoint(done, "gan", generator, discriminator, g_optimizer, d_optimizer, ckpt_hist, path=CKPT_PATH)
            remaining -= run

    # --------------------------- Guardado final ------------------------------
    os.makedirs('exports', exist_ok=True)

    # 1) Guarda state_dict (siempre)
    torch.save({'state_dict': generator.state_dict(), 'scale': 4},
               'exports/generator_state.pth')
    print("✅ Guardado state_dict → exports/generator_state.pth")

    # 2) Construye una instancia LIMPIA para exportar en CPU (evita deepcopy)
    #    Nota: si tu clase Generator requiere args, colócalos aquí.
    try:
        exp = Generator().eval().to('cpu')
        exp.load_state_dict(generator.state_dict(), strict=True)
    except Exception as e:
        print(f"⚠️ No pude clonar en una instancia limpia (state_dict). Error: {e}")
        # Último recurso: clonar “superficial” y mandar a CPU (menos recomendable)
        import copy
        exp = copy.deepcopy(generator).eval().to('cpu')

    # 3) Exporta TorchScript
    ok_ts = False
    example = torch.randn(1, 3, 64, 64)  # LR 64x64 → HR 256x256 para x4
    try:
        #scripted = torch.jit.trace(exp, example, check_trace=False)
        #scripted.save('exports/generator_script.pt')
        ok_ts = True
        print("✅ TorchScript (trace) → exports/generator_script.pt")
    except Exception as e_trace:
        print(f"⚠️ trace falló: {e_trace}")
        # Fallback: intenta script (puede fallar con módulos Python puros)
        try:
            scripted = torch.jit.script(exp)
            scripted.save('exports/generator_script.pt')
            ok_ts = True
            print("✅ TorchScript (script) → exports/generator_script.pt")
        except Exception as e_script:
            print(f"⚠️ script falló: {e_script}")

    # 4) Fallback ONNX (opcional, solo si necesitas un formato portable)
    if not ok_ts:
        try:
            torch.onnx.export(
                exp, example, 'exports/generator_export.onnx',
                input_names=['lr'], output_names=['sr'],
                opset_version=17, do_constant_folding=True,
                dynamic_axes={'lr': {0: 'batch'}, 'sr': {0: 'batch'}}
            )
            print("✅ ONNX → exports/generator_export.onnx")
        except Exception as e_onnx:
            print(f"❌ ONNX también falló: {e_onnx}")
            print("Sugerencias: 1) Asegura exp.forward puramente tensorial; 2) exporta con un tamaño de ejemplo compatible; 3) revisa KAN_Conv2d.")

    torch.save(discriminator.state_dict(), "discriminator_final.pth")
    ckpt_hist["srresnet"] = srresnet_history
    ckpt_hist["srgan"]    = srgan_history
    save_checkpoint(PRE_EPOCHS + GAN_EPOCHS, "final", generator, discriminator, g_optimizer, d_optimizer, ckpt_hist, path=CKPT_PATH)

    # Limpieza
    try:
        del exp, scripted
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()


    # -------------------------- Visualización final --------------------------
    try:
        plot_training_history(srgan_history, tag="srgan")
    except Exception as e:
        print(f"[plot_training_history] fallback: {e}")
        import matplotlib.pyplot as plt
        keys = [k for k in ("gen_loss","disc_loss","psnr") if k in srgan_history]
        for k in keys:
            plt.figure()
            plt.plot(srgan_history[k], label=k)
            plt.title(k); plt.xlabel("epoch"); plt.ylabel(k); plt.legend(); plt.show()

    # Muestras finales
    save_examples(generator, val_loader, device, epoch_label="final", max_batches=1)

# ---- entrypoint ----
if __name__ == "__main__":
    main()