from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def _import_mlflow():
    """
    Importación diferida para que MLflow siga siendo opcional.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow fue solicitado con --mlflow, pero no está "
            "instalado.\nInstálalo con:\n"
            "    python -m pip install mlflow"
        ) from exc

    return mlflow


def get_git_commit(project_root: Path) -> str:
    """
    Obtiene el commit actual de Git.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_dirty(project_root: Path) -> bool | None:
    """
    Indica si existen cambios sin commit.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def sha256_file(path: Path) -> str:
    """
    SHA-256 para identificar inequívocamente el checkpoint.
    """
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def start_benchmark_run(
    *,
    experiment_name: str,
    run_name: str,
):
    mlflow = _import_mlflow()

    mlflow.set_experiment(experiment_name)

    return mlflow.start_run(
        run_name=run_name,
    )


def log_benchmark_params(
    *,
    project_root: Path,
    checkpoint: Path,
    datasets: list[str],
    scale: int,
    shave_border: int,
    amp: bool,
    tile_size: int,
    tile_overlap: int,
    device: str,
    degradation_modes: dict[str, str],
):
    mlflow = _import_mlflow()

    checkpoint = checkpoint.resolve()

    params: dict[str, Any] = {
        "model": "SRGAN-CKAN",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "datasets": ",".join(datasets),
        "scale": scale,
        "shave_border": shave_border,
        "amp": amp,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "device": device,
        "git_commit": get_git_commit(project_root),
    }

    dirty = get_git_dirty(project_root)

    if dirty is not None:
        params["git_dirty"] = dirty

    for dataset_name, mode in degradation_modes.items():
        params[
            f"degradation_{dataset_name}"
        ] = mode

    mlflow.log_params(params)


def log_dataset_metrics(
    dataset_name: str,
    summary: dict,
):
    """
    Registra las métricas agregadas.

    El nombre del dataset forma parte de la clave para permitir
    múltiples benchmarks dentro de un mismo run.
    """
    mlflow = _import_mlflow()

    metric_names = (
        "psnr_y",
        "psnr_rgb",
        "ms_ssim_y",
        "ms_ssim_rgb",
        "lpips",
        "bicubic_psnr_y",
        "bicubic_ms_ssim_y",
        "bicubic_lpips",
    )

    for metric_name in metric_names:

        value = summary.get(metric_name)

        if value is None:
            continue

        mlflow.log_metric(
            f"{dataset_name}.{metric_name}",
            float(value),
        )

    mlflow.log_metric(
        f"{dataset_name}.images",
        float(summary["images"]),
    )


def log_benchmark_artifacts(
    output_dir: Path,
):
    mlflow = _import_mlflow()

    artifact_files = (
        "metrics_per_image.csv",
        "summary.csv",
        "benchmark_config.json",
    )

    for filename in artifact_files:

        path = output_dir / filename

        if path.exists():
            mlflow.log_artifact(
                str(path),
                artifact_path="benchmark",
            )