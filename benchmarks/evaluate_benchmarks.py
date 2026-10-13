from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image


# ---------------------------------------------------------
# Raíz del proyecto
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Imports del proyecto
# ---------------------------------------------------------

from benchmarks.benchmark_dataset import BenchmarkDataset

from benchmarks.mlflow_tracking import (
    log_benchmark_artifacts,
    log_benchmark_params,
    log_dataset_metrics,
    start_benchmark_run,
)

# Generator.py utiliza nn.KAN_Conv2d, por lo que CKAN debe
# registrarse antes de importar/instanciar Generator.
from KAN.KAN_Conv2d import KAN_Conv2d

if not hasattr(nn, "KAN_Conv2d"):
    nn.KAN_Conv2d = KAN_Conv2d

from models.Generator import Generator


# ---------------------------------------------------------
# Configuración
# ---------------------------------------------------------

SUPPORTED_DATASETS = {
    "Set5": {
        "hr": PROJECT_ROOT / "dataset/benchmarks/Set5/HR",
        "lr": PROJECT_ROOT / "dataset/benchmarks/Set5/LR_x4",
    },
    "Set14": {
        "hr": PROJECT_ROOT / "dataset/benchmarks/Set14/HR",
        "lr": PROJECT_ROOT / "dataset/benchmarks/Set14/LR_x4",
    },
    "BSD100": {
        "hr": PROJECT_ROOT / "dataset/benchmarks/BSD100/HR",
        "lr": PROJECT_ROOT / "dataset/benchmarks/BSD100/LR_x4",
    },
    "Urban100": {
        "hr": PROJECT_ROOT / "dataset/benchmarks/Urban100/HR",
        "lr": PROJECT_ROOT / "dataset/benchmarks/Urban100/LR_x4",
    },
    "Manga109": {
        "hr": PROJECT_ROOT / "dataset/benchmarks/Manga109/HR",
        "lr": PROJECT_ROOT / "dataset/benchmarks/Manga109/LR_x4",
    },
}

# =========================================================
# Utilidades
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark estandarizado para SRGAN-CKAN."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta al checkpoint del generador.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=(
            "Datasets a evaluar. "
            "Ejemplo: --datasets Set5 Set14 Urban100 "
            "o --datasets all"
        ),
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Factor de super-resolución.",
    )

    parser.add_argument(
        "--shave-border",
        type=int,
        default=None,
        help=(
            "Píxeles eliminados en cada borde antes de calcular "
            "las métricas. Por defecto = scale."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmarks/srgan_ckan",
        help="Directorio donde guardar resultados.",
    )

    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Guardar HR, LR bicubic y SR.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cpu, cuda:0, etc.",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Usar automatic mixed precision en CUDA.",
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help=(
            "Tamaño del tile LR para inferencia por bloques. "
            "0 = imagen completa."
        ),
    )

    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=16,
        help="Solapamiento entre tiles LR.",
    )

    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Registrar la evaluación como un run de MLflow.",
    )

    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="SRGAN-CKAN-Benchmarks",
        help="Nombre del experimento de MLflow.",
    )

    parser.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help=(
            "Nombre opcional del run. "
            "Por defecto se genera a partir del checkpoint y datasets."
        ),
    )

    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

def resolve_degradation_mode(
    dataset_config: dict,
) -> tuple[str, Path | None]:

    lr_root = dataset_config["lr"]

    has_official_lr = (
        lr_root.exists()
        and any(
            p.is_file()
            and p.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff",
            }
            for p in lr_root.iterdir()
        )
    )

    if has_official_lr:
        return "official", lr_root

    return "bicubic", None

def resolve_datasets(dataset_args: List[str]) -> List[str]:
    if len(dataset_args) == 1 and dataset_args[0].lower() == "all":
        return list(SUPPORTED_DATASETS.keys())

    unknown = [
        name for name in dataset_args
        if name not in SUPPORTED_DATASETS
    ]

    if unknown:
        raise ValueError(
            f"Datasets no soportados: {unknown}\n"
            f"Disponibles: {list(SUPPORTED_DATASETS.keys())}"
        )

    return dataset_args


# =========================================================
# Checkpoint
# =========================================================


def load_generator(
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Carga el Generator.

    Esta función soporta varios formatos comunes de checkpoint:
        - state_dict directamente
        - {"generator": state_dict}
        - {"generator_state_dict": state_dict}
        - {"model_state_dict": state_dict}
        - {"state_dict": state_dict}

    Si tu checkpoint.py utiliza otro formato, ésta es la única
    función que habría que modificar.
    """

    model = Generator()

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = None

    if isinstance(checkpoint, dict):

        possible_keys = (
            "generator_state_dict",
            "generator",
            "model_state_dict",
            "state_dict",
        )

        for key in possible_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

        # Puede ser directamente un state_dict.
        if state_dict is None:
            if all(
                isinstance(key, str)
                for key in checkpoint.keys()
            ):
                state_dict = checkpoint

    else:
        state_dict = checkpoint

    if state_dict is None:
        raise RuntimeError(
            "No fue posible localizar el state_dict "
            "del generador en el checkpoint."
        )

    # Eliminar prefijo module. si proviene de DataParallel.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model = model.to(device)

    if device.type == "cuda":
        model = model.to(
            memory_format=torch.channels_last
        )

    model.eval()

    return model


# =========================================================
# Conversión RGB -> Y
# =========================================================


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """
    Convierte RGB [0,1] a luminancia Y.

    Espera:
        [N, 3, H, W]

    Devuelve:
        [N, 1, H, W]

    Se utiliza una conversión estándar basada en BT.601.
    """

    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(
            f"Se esperaba tensor [N,3,H,W], recibido {x.shape}"
        )

    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]

    y = (
        16.0
        + 65.481 * r
        + 128.553 * g
        + 24.966 * b
    ) / 255.0

    return y


# =========================================================
# Border shave
# =========================================================


def shave_border(
    x: torch.Tensor,
    border: int,
) -> torch.Tensor:

    if border <= 0:
        return x

    height = x.shape[-2]
    width = x.shape[-1]

    if height <= 2 * border or width <= 2 * border:
        raise ValueError(
            f"Imagen demasiado pequeña para shave={border}: "
            f"{height}x{width}"
        )

    return x[
        ...,
        border:-border,
        border:-border,
    ]


# =========================================================
# PSNR
# =========================================================


def calculate_psnr(
    sr: torch.Tensor,
    hr: torch.Tensor,
) -> float:

    mse = F.mse_loss(
        sr,
        hr,
        reduction="mean",
    ).item()

    if mse == 0:
        return float("inf")

    return 10.0 * math.log10(1.0 / mse)


# =========================================================
# SSIM / MS-SSIM
# =========================================================


def get_ms_ssim_function():
    """
    Intenta utilizar pytorch_msssim.

    El proyecto ya utiliza MS-SSIM; si el paquete está
    disponible, lo aprovechamos directamente.
    """

    try:
        from pytorch_msssim import ms_ssim

        return ms_ssim

    except ImportError:
        return None


def calculate_ms_ssim(
    sr: torch.Tensor,
    hr: torch.Tensor,
) -> float | None:

    ms_ssim = get_ms_ssim_function()

    if ms_ssim is None:
        return None

    value = ms_ssim(
        sr,
        hr,
        data_range=1.0,
        size_average=True,
    )

    return float(value.item())


# =========================================================
# LPIPS
# =========================================================


class LPIPSMetric:

    def __init__(
        self,
        device: torch.device,
    ):
        self.device = device
        self.metric = None

        try:
            import lpips

            self.metric = lpips.LPIPS(
                net="alex"
            ).to(device)

            self.metric.eval()

        except ImportError:
            self.metric = None

    @torch.no_grad()
    def __call__(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
    ) -> float | None:

        if self.metric is None:
            return None

        # LPIPS espera [-1, 1].
        sr_lpips = sr * 2.0 - 1.0
        hr_lpips = hr * 2.0 - 1.0

        value = self.metric(
            sr_lpips,
            hr_lpips,
        )

        return float(value.mean().item())


# =========================================================
# Inferencia
# =========================================================


@torch.no_grad()
def infer_full_image(
    model: torch.nn.Module,
    lr: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:

    lr = lr.to(device)

    if device.type == "cuda":
        lr = lr.to(
            memory_format=torch.channels_last
        )

    autocast_enabled = (
        use_amp and device.type == "cuda"
    )

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=autocast_enabled,
    ):
        sr = model(lr)

    return sr.float().clamp(0.0, 1.0)


@torch.no_grad()
def infer_tiled(
    model: torch.nn.Module,
    lr: torch.Tensor,
    device: torch.device,
    scale: int,
    tile_size: int,
    overlap: int,
    use_amp: bool,
) -> torch.Tensor:
    """
    Inferencia por tiles.

    El tile_size corresponde al tamaño sobre LR.
    """

    if tile_size <= 0:
        return infer_full_image(
            model,
            lr,
            device,
            use_amp,
        )

    if lr.shape[0] != 1:
        raise ValueError(
            "La inferencia tiled actualmente requiere batch=1."
        )

    _, channels, height, width = lr.shape

    stride = tile_size - overlap

    if stride <= 0:
        raise ValueError(
            "tile-overlap debe ser menor que tile-size."
        )

    sr_height = height * scale
    sr_width = width * scale

    output = torch.zeros(
        (
            1,
            channels,
            sr_height,
            sr_width,
        ),
        dtype=torch.float32,
        device=device,
    )

    weight = torch.zeros_like(output)

    y_positions = list(
        range(0, max(height - tile_size, 0) + 1, stride)
    )

    x_positions = list(
        range(0, max(width - tile_size, 0) + 1, stride)
    )

    if not y_positions or y_positions[-1] != max(
        height - tile_size,
        0,
    ):
        y_positions.append(
            max(height - tile_size, 0)
        )

    if not x_positions or x_positions[-1] != max(
        width - tile_size,
        0,
    ):
        x_positions.append(
            max(width - tile_size, 0)
        )

    for y in y_positions:
        for x in x_positions:

            tile = lr[
                :,
                :,
                y:y + tile_size,
                x:x + tile_size,
            ]

            sr_tile = infer_full_image(
                model,
                tile,
                device,
                use_amp,
            )

            y_sr = y * scale
            x_sr = x * scale

            tile_h_sr = sr_tile.shape[-2]
            tile_w_sr = sr_tile.shape[-1]

            output[
                :,
                :,
                y_sr:y_sr + tile_h_sr,
                x_sr:x_sr + tile_w_sr,
            ] += sr_tile

            weight[
                :,
                :,
                y_sr:y_sr + tile_h_sr,
                x_sr:x_sr + tile_w_sr,
            ] += 1.0

    output = output / weight.clamp_min(1.0)

    return output.clamp(0.0, 1.0)


# =========================================================
# Bicubic baseline
# =========================================================


def bicubic_upsample(
    lr: torch.Tensor,
    scale: int,
) -> torch.Tensor:

    return F.interpolate(
        lr,
        scale_factor=scale,
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)


# =========================================================
# Guardado
# =========================================================


def save_tensor_image(
    tensor: torch.Tensor,
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = tensor.detach().cpu()

    if image.ndim == 4:
        image = image.squeeze(0)

    image = image.clamp(0.0, 1.0)

    to_pil_image(image).save(path)


def write_csv(
    path: Path,
    rows: List[dict],
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# Evaluación
# =========================================================


def evaluate_dataset(
    dataset_name: str,
    dataset_config: dict,
    model: torch.nn.Module,
    device: torch.device,
    scale: int,
    shave: int,
    output_dir: Path,
    save_images: bool,
    use_amp: bool,
    tile_size: int,
    tile_overlap: int,
    lpips_metric: LPIPSMetric,
):
    hr_root = dataset_config["hr"]
    lr_root = dataset_config["lr"]

    degradation_mode, resolved_lr_root = (
        resolve_degradation_mode(
            dataset_config
        )
    )

    dataset = BenchmarkDataset(
        root=hr_root,
        lr_root=resolved_lr_root,
        scale=scale,
    )

    if dataset.mode != degradation_mode:
        raise RuntimeError(
            "Inconsistencia en el protocolo de degradación: "
            f"resolver={degradation_mode}, "
            f"dataset={dataset.mode}"
        )

    print()
    print(f"Degradation : {dataset.mode}")
    print(f"HR           : {hr_root}")

    if dataset.mode == "official":
        print(f"LR           : {lr_root}")
    else:
        print("LR           : generated bicubic")

    print()
    print("=" * 72)
    print(
        f"{dataset_name}: {len(dataset)} imágenes"
    )
    print("=" * 72)

    rows = []

    for index in range(len(dataset)):

        sample = dataset[index]

        name = sample["name"]

        hr = sample["hr"].unsqueeze(0)
        lr = sample["lr"].unsqueeze(0)

        hr = hr.to(device)

        # -------------------------------------------------
        # Inferencia
        # -------------------------------------------------

        sr = infer_tiled(
            model=model,
            lr=lr,
            device=device,
            scale=scale,
            tile_size=tile_size,
            overlap=tile_overlap,
            use_amp=use_amp,
        )

        # Ajuste defensivo.
        target_h = hr.shape[-2]
        target_w = hr.shape[-1]

        sr = sr[
            ...,
            :target_h,
            :target_w,
        ]

        # -------------------------------------------------
        # Baseline bicubic
        # -------------------------------------------------

        bicubic = bicubic_upsample(
            lr.to(device),
            scale=scale,
        )

        bicubic = bicubic[
            ...,
            :target_h,
            :target_w,
        ]

        # -------------------------------------------------
        # Border shave
        # -------------------------------------------------

        sr_eval = shave_border(
            sr,
            shave,
        )

        hr_eval = shave_border(
            hr,
            shave,
        )

        bicubic_eval = shave_border(
            bicubic,
            shave,
        )

        # -------------------------------------------------
        # Y
        # -------------------------------------------------

        sr_y = rgb_to_y(sr_eval)
        hr_y = rgb_to_y(hr_eval)

        bicubic_y = rgb_to_y(
            bicubic_eval
        )

        # -------------------------------------------------
        # Métricas SR
        # -------------------------------------------------

        psnr_y = calculate_psnr(
            sr_y,
            hr_y,
        )

        psnr_rgb = calculate_psnr(
            sr_eval,
            hr_eval,
        )

        ms_ssim_y = calculate_ms_ssim(
            sr_y,
            hr_y,
        )

        ms_ssim_rgb = calculate_ms_ssim(
            sr_eval,
            hr_eval,
        )

        lpips_value = lpips_metric(
            sr_eval,
            hr_eval,
        )

        # -------------------------------------------------
        # Métricas Bicubic
        # -------------------------------------------------

        bicubic_psnr_y = calculate_psnr(
            bicubic_y,
            hr_y,
        )

        bicubic_ms_ssim_y = calculate_ms_ssim(
            bicubic_y,
            hr_y,
        )

        bicubic_lpips = lpips_metric(
            bicubic_eval,
            hr_eval,
        )

        row = {
            "dataset": dataset_name,
            "image": name,

            "psnr_y": psnr_y,
            "psnr_rgb": psnr_rgb,

            "ms_ssim_y": ms_ssim_y,
            "ms_ssim_rgb": ms_ssim_rgb,

            "lpips": lpips_value,

            "bicubic_psnr_y": bicubic_psnr_y,
            "bicubic_ms_ssim_y": bicubic_ms_ssim_y,
            "bicubic_lpips": bicubic_lpips,

            "height": target_h,
            "width": target_w,
        }

        rows.append(row)

        print(
            f"[{index + 1:03d}/{len(dataset):03d}] "
            f"{name:<30} "
            f"PSNR-Y={psnr_y:7.3f} "
            f"MS-SSIM-Y="
            f"{ms_ssim_y if ms_ssim_y is not None else 'N/A'} "
            f"LPIPS="
            f"{lpips_value if lpips_value is not None else 'N/A'}"
        )

        # -------------------------------------------------
        # Guardar imágenes
        # -------------------------------------------------

        if save_images:

            image_dir = (
                output_dir
                / dataset_name
                / name
            )

            save_tensor_image(
                hr,
                image_dir / "HR.png",
            )

            save_tensor_image(
                bicubic,
                image_dir / "Bicubic.png",
            )

            save_tensor_image(
                sr,
                image_dir / "SR.png",
            )

            # LR se amplía únicamente para visualizarla.
            lr_visual = bicubic_upsample(
                lr,
                scale,
            )

            save_tensor_image(
                lr_visual,
                image_dir / "LR_visual.png",
            )

    return rows


# =========================================================
# Resumen
# =========================================================


def mean_optional(
    values: List[float | None],
) -> float | None:

    valid = [
        value
        for value in values
        if value is not None
        and not math.isnan(value)
    ]

    if not valid:
        return None

    return sum(valid) / len(valid)


def summarize_dataset(
    dataset_name: str,
    rows: List[dict],
) -> dict:

    return {
        "dataset": dataset_name,
        "images": len(rows),

        "psnr_y":
            mean_optional([
                row["psnr_y"]
                for row in rows
            ]),

        "psnr_rgb":
            mean_optional([
                row["psnr_rgb"]
                for row in rows
            ]),

        "ms_ssim_y":
            mean_optional([
                row["ms_ssim_y"]
                for row in rows
            ]),

        "ms_ssim_rgb":
            mean_optional([
                row["ms_ssim_rgb"]
                for row in rows
            ]),

        "lpips":
            mean_optional([
                row["lpips"]
                for row in rows
            ]),

        "bicubic_psnr_y":
            mean_optional([
                row["bicubic_psnr_y"]
                for row in rows
            ]),

        "bicubic_ms_ssim_y":
            mean_optional([
                row["bicubic_ms_ssim_y"]
                for row in rows
            ]),

        "bicubic_lpips":
            mean_optional([
                row["bicubic_lpips"]
                for row in rows
            ]),
    }


def print_summary(
    summaries: List[dict],
):

    print()
    print("=" * 92)
    print("BENCHMARK SUMMARY")
    print("=" * 92)

    print(
        f"{'Dataset':<14}"
        f"{'Images':>8}"
        f"{'PSNR-Y':>12}"
        f"{'MS-SSIM-Y':>14}"
        f"{'LPIPS':>12}"
        f"{'Bicubic PSNR':>16}"
    )

    print("-" * 92)

    for summary in summaries:

        ms_ssim = summary["ms_ssim_y"]
        lpips_value = summary["lpips"]

        print(
            f"{summary['dataset']:<14}"
            f"{summary['images']:>8}"
            f"{summary['psnr_y']:>12.4f}"
            f"{ms_ssim if ms_ssim is not None else float('nan'):>14.5f}"
            f"{lpips_value if lpips_value is not None else float('nan'):>12.5f}"
            f"{summary['bicubic_psnr_y']:>16.4f}"
        )

    print("=" * 92)


# =========================================================
# Main
# =========================================================


def main():

    args = parse_args()

    device = resolve_device(
        args.device
    )

    datasets = resolve_datasets(
        args.datasets
    )

    shave = (
        args.scale
        if args.shave_border is None
        else args.shave_border
    )

    output_dir = (
        PROJECT_ROOT
        / args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SRGAN-CKAN Benchmark Evaluation")
    print("=" * 72)

    print(
        f"Device       : {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(
        f"Scale        : x{args.scale}"
    )

    print(
        f"Border shave : {shave}px"
    )

    print(
        f"AMP          : {args.amp}"
    )

    print(
        f"Tile size    : "
        f"{args.tile_size if args.tile_size else 'disabled'}"
    )

    print(
        f"Datasets     : "
        f"{', '.join(datasets)}"
    )

    print(
        f"Checkpoint   : {args.checkpoint}"
    )

    # -----------------------------------------------------
    # Modelo
    # -----------------------------------------------------

    print()
    print("Cargando Generator...")

    model = load_generator(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    print("Generator cargado.")

    # -----------------------------------------------------
    # Métrica LPIPS
    # -----------------------------------------------------

    lpips_metric = LPIPSMetric(
        device=device
    )

    if lpips_metric.metric is None:
        print(
            "WARNING: paquete lpips no encontrado. "
            "LPIPS no será calculado."
        )

    if get_ms_ssim_function() is None:
        print(
            "WARNING: pytorch-msssim no encontrado. "
            "MS-SSIM no será calculado."
        )

    # -----------------------------------------------------
    # Evaluación
    # -----------------------------------------------------

    all_rows = []
    summaries = []
    degradation_modes = {}

    for dataset_name in datasets:
        dataset_config = (
            SUPPORTED_DATASETS[
                dataset_name
            ]
        )

        rows = evaluate_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            model=model,
            device=device,
            scale=args.scale,
            shave=shave,
            output_dir=output_dir,
            save_images=args.save_images,
            use_amp=args.amp,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            lpips_metric=lpips_metric,
        )

        degradation_mode, _ = resolve_degradation_mode(
            dataset_config
        )

        degradation_modes[dataset_name] = (
            degradation_mode
        )

        all_rows.extend(rows)

        summary = summarize_dataset(
            dataset_name,
            rows,
        )

        summaries.append(summary)

    # -----------------------------------------------------
    # Guardado de CSV
    # -----------------------------------------------------

    write_csv(
        output_dir
        / "metrics_per_image.csv",
        all_rows,
    )

    write_csv(
        output_dir
        / "summary.csv",
        summaries,
    )

    # -----------------------------------------------------
    # JSON de configuración
    # -----------------------------------------------------

    config = {
        "checkpoint": args.checkpoint,
        "datasets": datasets,
        "scale": args.scale,
        "shave_border": shave,
        "amp": args.amp,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "device": str(device),
        "degradation": degradation_modes,
    }

    with (
        output_dir / "benchmark_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # MLflow
    # -----------------------------------------------------

    if args.mlflow:

        checkpoint_path = Path(args.checkpoint)

        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path

        run_name = args.mlflow_run_name

        if run_name is None:
            checkpoint_name = checkpoint_path.stem
            dataset_label = "-".join(datasets)

            run_name = (
                f"{checkpoint_name}-"
                f"{dataset_label}-"
                f"x{args.scale}"
            )

        with start_benchmark_run(
                experiment_name=args.mlflow_experiment,
                run_name=run_name,
        ):

            log_benchmark_params(
                project_root=PROJECT_ROOT,
                checkpoint=checkpoint_path,
                datasets=datasets,
                scale=args.scale,
                shave_border=shave,
                amp=args.amp,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                device=str(device),
                degradation_modes=degradation_modes,
            )

            for summary in summaries:
                log_dataset_metrics(
                    summary["dataset"],
                    summary,
                )

            log_benchmark_artifacts(
                output_dir,
            )

        print()
        print(
            f"MLflow run registrado: {run_name}"
        )

    # -----------------------------------------------------
    # Resumen
    # -----------------------------------------------------

    print_summary(
        summaries
    )

    print()
    print(
        f"Resultados guardados en:"
    )

    print(
        output_dir.resolve()
    )


if __name__ == "__main__":
    main()