import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from KAN._KANProjector import _KANProjector



class KAN_Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias: bool = True,
                 chunk_pixels: int = 8192):
        super().__init__()
        # Normaliza a pares 2D
        self.kernel_size = _pair(kernel_size)   # (kH, kW)
        self.stride      = _pair(stride)        # (sH, sW)
        self.padding     = _pair(padding)       # (pH, pW)
        self.dilation    = _pair(dilation)      # (dH, dW)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.bias_flag    = bias
        self.chunk_pixels = int(chunk_pixels)

        # *** proyector por defecto ***
        kH, kW = self.kernel_size
        in_col_channels = in_channels * kH * kW
        self.KAN_NM = _KANProjector(in_col_channels, out_channels, bias=bias)

        # Usa tu proyección KAN existente
        # Debes haber un módulo que mapee col (B, C*kH*kW, L) -> (B, out_channels, L)
        # se asume como self.KAN_NM
        try:
            self.KAN_NM  # verificar que exista si se inyecta desde fuera
        except AttributeError:
            # Coloca aquí tu implementación/constructor real de KAN_NM si corresponde
            # raise si no está definido para hacer visible el problema
            raise AttributeError("KAN_Conv2d requiere un submódulo `self.KAN_NM` que proyecte col -> (B, out, L).")

    # --- Helpers de salida 2D exactos ---
    @staticmethod
    def _calc_out_dim(L_in, pad, dil, k, stride):
        # Fórmula conv2d estándar (PyTorch): floor((L_in + 2p - d*(k-1) - 1)/s + 1)
        return (L_in + 2*pad - dil*(k - 1) - 1) // stride + 1

    def _im2col_band(self, x, row_start, row_end, kH, kW, sH, sW, pH, pW, dH, dW):
        """
        Devuelve una banda segura para unfold con padding vertical aplicado explícitamente.
        - row_start, row_end: filas de salida [row_start, row_end) que queremos producir.
        - Aplica F.pad en vertical para compensar el recorte.
        - Devuelve col_band (B, C*kH*kW, L_band) y band_W_out.
        """
        B, C, H, W = x.shape

        # Receptive field vertical (antes de recorte a [0,H))
        top_unclipped = row_start * sH - pH
        bot_unclipped = (row_end - 1) * sH - pH + (kH - 1) * dH + 1  # exclusivo

        # Recorte a la imagen
        in_row_start = max(top_unclipped, 0)
        in_row_end   = min(bot_unclipped, H)

        # Recorte mínimo de entrada
        x_cut = x[:, :, int(in_row_start):int(in_row_end), :]  # (B, C, H_cut, W)
        H_cut = x_cut.shape[2]

        # ¿Cuánto padding vertical falta por lo que recortamos?
        top_pad_needed    = int(max(0, 0 - top_unclipped))         # si top_unclipped < 0
        bottom_pad_needed = int(max(0, bot_unclipped - H))         # si bot_unclipped > H

        # Aplica padding vertical explícito a la banda
        # F.pad usa orden (left, right, top, bottom)
        if top_pad_needed or bottom_pad_needed:
            x_cut = F.pad(x_cut, (0, 0, top_pad_needed, bottom_pad_needed))

        # Ahora aplica unfold SOLO con padding horizontal (pW)
        col_band = F.unfold(
            x_cut,
            kernel_size=(kH, kW),
            dilation=(dH, dW),
            padding=(0, pW),          # vertical ya se compensó arriba
            stride=(sH, sW)
        )  # (B, C*kH*kW, L_band)

        # W_out dentro de la banda es el mismo que global (recortamos solo en vertical)
        band_W_out = self._calc_out_dim(W, pW, dW, kW, sW)

        # Validaciones fuertes: que la banda produzca al menos 1 fila
        L_band = col_band.shape[-1]
        if band_W_out <= 0 or L_band == 0 or (L_band % band_W_out) != 0:
            raise RuntimeError(
                f"Unfold banda inválida: L_band={L_band}, band_W_out={band_W_out}, "
                f"H_cut={x_cut.shape[2]}, pads=({top_pad_needed},{bottom_pad_needed}), "
                f"params k=({kH},{kW}) s=({sH},{sW}) p=({pH},{pW}) d=({dH},{dW})"
            )

        return col_band, band_W_out


    # Usa la altura real que produce F.unfold en la banda (band_H_out_actual)
    # en lugar de asumir row_end - row_start. Además avanza 'row' con esa altura real.

    def forward(self, x, update_grid: bool = False):
        B, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation

        # Tamaños de salida globales
        H_out = self._calc_out_dim(H, pH, dH, kH, sH)
        W_out = self._calc_out_dim(W, pW, dW, kW, sW)
        L = H_out * W_out

        # Ruta rápida si es pequeño
        if L <= self.chunk_pixels:
            col = F.unfold(
                x,
                kernel_size=(kH, kW),
                dilation=(dH, dW),
                padding=(pH, pW),
                stride=(sH, sW)
            )  # (B, C*kH*kW, L)
            out = self.KAN_NM(col, update_grid=update_grid)  # (B, out_channels, L)
            return out.view(B, self.out_channels, H_out, W_out)

        # Ruta chunked por bandas de filas
        outputs = []
        row = 0
        per_row = W_out
        max_rows_per_chunk = max(1, (self.chunk_pixels // per_row))  # fuerza múltiplos de W_out

        while row < H_out:
            row_end_plan = min(row + max_rows_per_chunk, H_out)

            # _im2col_band AHORA retorna: (col_band, band_W_out)
            col_band, band_W_out = self._im2col_band(
                x, row, row_end_plan, kH, kW, sH, sW, pH, pW, dH, dW
            )  # (B, C*kH*kW, L_band), band_W_out == W_out

            L_band = col_band.shape[-1]
            band_H_out_actual = L_band // band_W_out
            if band_H_out_actual * band_W_out != L_band or band_H_out_actual <= 0:
                raise RuntimeError(
                    f"Band shape mismatch: L_band={L_band}, band_W_out={band_W_out}, "
                    f"band_H_out_actual={band_H_out_actual}"
                )

            out_band = self.KAN_NM(col_band, update_grid=update_grid)  # (B, out, L_band)
            out_band = out_band.view(B, self.out_channels, band_H_out_actual, band_W_out)

            outputs.append(out_band)
            row += band_H_out_actual

        out_full = torch.cat(outputs, dim=2)  # (B, out, H_out_concat, W_out)

        # Alinea exactamente a H_out (por seguridad ante redondeos)
        if out_full.shape[2] > H_out:
            out_full = out_full[:, :, :H_out, :]
        elif out_full.shape[2] < H_out:
            pad_rows = H_out - out_full.shape[2]
            out_full = F.pad(out_full, (0, 0, 0, pad_rows))

        return out_full

        # Si por redondeos se pasó por 1 fila, recorta o pad a H_out exacto
        if out_full.shape[2] > H_out:
            out_full = out_full[:, :, :H_out, :]
        elif out_full.shape[2] < H_out:
            # pad con ceros (raro; sólo si una fila no se generó)
            pad_rows = H_out - out_full.shape[2]
            out_full = F.pad(out_full, (0, 0, 0, pad_rows))

        return out_full


setattr(nn, "KAN_Conv2d", KAN_Conv2d)