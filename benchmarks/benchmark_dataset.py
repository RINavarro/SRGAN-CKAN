from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


class BenchmarkDataset(Dataset):
    """
    Dataset para evaluación estandarizada de Super-Resolution.

    Soporta dos modos:

    1. Bicubic:
       Sólo se proporciona HR. El LR se genera mediante downsampling
       bicúbico con el factor `scale`.

    2. Official:
       Se proporcionan directorios HR y LR. Los pares son validados
       estrictamente antes de comenzar la evaluación.
    """

    def __init__(
        self,
        root,
        scale=4,
        lr_root=None,
    ):
        self.root = Path(root)
        self.scale = scale
        self.lr_root = Path(lr_root) if lr_root is not None else None

        if not self.root.exists():
            raise FileNotFoundError(
                f"No existe el directorio HR: {self.root}"
            )

        self.hr_images = sorted(
            p for p in self.root.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not self.hr_images:
            raise RuntimeError(
                f"No se encontraron imágenes en {self.root}"
            )

        # ---------------------------------------------------------
        # Modo bicubic
        # ---------------------------------------------------------
        if self.lr_root is None:
            self.mode = "bicubic"
            self.samples = [
                (hr_path, None)
                for hr_path in self.hr_images
            ]

        # ---------------------------------------------------------
        # Modo official HR/LR
        # ---------------------------------------------------------
        else:
            self.mode = "official"

            if not self.lr_root.exists():
                raise FileNotFoundError(
                    f"No existe el directorio LR: {self.lr_root}"
                )

            lr_images = sorted(
                p for p in self.lr_root.iterdir()
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )

            if not lr_images:
                raise RuntimeError(
                    f"No se encontraron imágenes LR en {self.lr_root}"
                )

            self.samples = self._build_official_pairs(
                self.hr_images,
                lr_images,
            )

    # -------------------------------------------------------------
    # Normalización del identificador de imagen
    # -------------------------------------------------------------
    def _image_id(self, path):
        """
        Convierte:

            img_001_SRF_4_HR.png
            img_001_SRF_4_LR.png

        en:

            img_001

        También funciona para nombres que no siguen SRF_x.
        """

        stem = path.stem

        hr_suffix = f"_SRF_{self.scale}_HR"
        lr_suffix = f"_SRF_{self.scale}_LR"

        if stem.endswith(hr_suffix):
            return stem[:-len(hr_suffix)]

        if stem.endswith(lr_suffix):
            return stem[:-len(lr_suffix)]

        if stem.endswith("_HR"):
            return stem[:-3]

        if stem.endswith("_LR"):
            return stem[:-3]

        return stem

    # -------------------------------------------------------------
    # Emparejamiento HR/LR
    # -------------------------------------------------------------
    def _build_official_pairs(self, hr_images, lr_images):

        hr_map = {
            self._image_id(path): path
            for path in hr_images
        }

        lr_map = {
            self._image_id(path): path
            for path in lr_images
        }

        hr_ids = set(hr_map)
        lr_ids = set(lr_map)

        missing_lr = sorted(hr_ids - lr_ids)
        missing_hr = sorted(lr_ids - hr_ids)

        if missing_lr:
            raise RuntimeError(
                "Faltan imágenes LR para:\n"
                + "\n".join(missing_lr)
            )

        if missing_hr:
            raise RuntimeError(
                "Existen imágenes LR sin HR correspondiente:\n"
                + "\n".join(missing_hr)
            )

        return [
            (hr_map[image_id], lr_map[image_id])
            for image_id in sorted(hr_ids)
        ]

    def __len__(self):
        return len(self.samples)

    # -------------------------------------------------------------
    # Lectura
    # -------------------------------------------------------------
    def __getitem__(self, index):

        hr_path, lr_path = self.samples[index]

        hr = Image.open(hr_path).convert("RGB")

        # =========================================================
        # OFFICIAL
        # =========================================================
        if self.mode == "official":

            lr = Image.open(lr_path).convert("RGB")

            hr_width, hr_height = hr.size
            lr_width, lr_height = lr.size

            expected_width = lr_width * self.scale
            expected_height = lr_height * self.scale

            if (
                hr_width != expected_width
                or hr_height != expected_height
            ):
                raise ValueError(
                    "\nDimensiones HR/LR incompatibles:\n"
                    f"  HR: {hr_path.name} "
                    f"{hr_width}x{hr_height}\n"
                    f"  LR: {lr_path.name} "
                    f"{lr_width}x{lr_height}\n"
                    f"  Escala esperada: x{self.scale}\n"
                    f"  HR esperado: "
                    f"{expected_width}x{expected_height}"
                )

            evaluation_size = (
                hr_height,
                hr_width,
            )

        # =========================================================
        # BICUBIC
        # =========================================================
        else:

            width, height = hr.size

            valid_width = width - (width % self.scale)
            valid_height = height - (height % self.scale)

            if valid_width == 0 or valid_height == 0:
                raise ValueError(
                    f"Imagen demasiado pequeña para x{self.scale}: "
                    f"{hr_path}"
                )

            hr = hr.crop(
                (0, 0, valid_width, valid_height)
            )

            lr = hr.resize(
                (
                    valid_width // self.scale,
                    valid_height // self.scale,
                ),
                resample=Image.Resampling.BICUBIC,
            )

            evaluation_size = (
                valid_height,
                valid_width,
            )

        # ---------------------------------------------------------
        # PIL -> Tensor [0, 1]
        # ---------------------------------------------------------
        hr_tensor = (
            pil_to_tensor(hr)
            .to(torch.float32)
            .div_(255.0)
        )

        lr_tensor = (
            pil_to_tensor(lr)
            .to(torch.float32)
            .div_(255.0)
        )

        return {
            "name": self._image_id(hr_path),
            "path": str(hr_path),
            "hr_path": str(hr_path),
            "lr_path": (
                str(lr_path)
                if lr_path is not None
                else None
            ),
            "hr": hr_tensor,
            "lr": lr_tensor,
            "degradation": self.mode,
            "original_size": (
                hr.height,
                hr.width,
            ),
            "evaluation_size": evaluation_size,
        }