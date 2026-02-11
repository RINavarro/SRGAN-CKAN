from torch import nn

from KAN.KANLinearNS_FactorizedLinear import KANLinearNS_FactorizedLinear


class KAN_NM(nn.Module):
    """
    layers_hidden: lista con dimensiones [in, h1, h2, ..., out]
    """
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, dropout=0.1):
        super().__init__()
        assert len(layers_hidden) >= 2, "layers_hidden debe contener al menos [in, out]"
        self.layers = nn.ModuleList()
        for i, (in_f, out_f) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            self.layers.append(
                KANLinearNS_FactorizedLinear(
                    in_f, out_f, grid_size=grid_size, spline_order=spline_order
                )
            )
            if i < len(layers_hidden) - 2:
                self.layers.append(nn.Dropout(dropout))  # no cambia complejidad asintótica

    def forward(self, x, update_grid: bool = False):
        for layer in self.layers:
            if isinstance(layer, KANLinearNS_FactorizedLinear):
                if update_grid:
                    layer.update_grid(x)
                x = layer(x)
            else:
                x = layer(x)
        return x