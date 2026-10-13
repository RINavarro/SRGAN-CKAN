import matplotlib.pyplot as plt

def save_training_figure(hist, out_path="training_curves_clean.png"):
    plt.style.use("default")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # (a)
    axs[0].plot(hist["gen_loss"], label="Generator", linewidth=2)
    axs[0].plot(hist["disc_loss"], label="Discriminator", linewidth=2)
    axs[0].set_title("(a) Generator vs Discriminator Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend()

    # (b)
    axs[1].plot(hist["content_loss"], label="Content", linewidth=2)
    axs[1].plot(hist["adv_loss"], label="Adversarial", linewidth=2)
    axs[1].set_title("(b) Content vs Adversarial Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend()

    # (c)
    axs[2].plot(hist["psnr"], label="PSNR (Y)", linewidth=2)
    axs[2].set_title("(c) PSNR Evolution")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("dB")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    axs[2].legend()

    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()