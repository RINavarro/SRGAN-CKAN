"""Pruebas de la integración convolucional CKAN y su interpretación matemática."""
import pytest
import torch
from torch.nn import functional as F
from KAN.KAN_Conv2d import KAN_Conv2d


@pytest.mark.parametrize('degree', [1, 2, 3, 4])
def test_output_shape_and_gradients(degree):
    model = KAN_Conv2d(2, 3, kernel_size=3, padding=1,
                       spline_degree=degree, grid_size=4,
                       projector_chunk_size=8, chunk_pixels=100)
    x = (torch.randn(2, 2, 5, 6) * 0.25).requires_grad_()
    y = model(x)
    assert y.shape == (2, 3, 5, 6)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.KAN_NM.spline_weight.grad is not None
    assert model.KAN_NM.spline_weight.grad.abs().sum() > 0


@pytest.mark.parametrize('kwargs', [
    dict(kernel_size=3, stride=1, padding=1),
    dict(kernel_size=3, stride=2, padding=1),
    dict(kernel_size=(2, 3), stride=(2, 1), padding=(1, 2), dilation=(2, 1)),
])
def test_full_vs_bands_and_gradients(kwargs):
    torch.manual_seed(17)
    full = KAN_Conv2d(2, 2, grid_size=3, spline_degree=2,
                      chunk_pixels=9999, projector_chunk_size=7, **kwargs)
    bands = KAN_Conv2d(2, 2, grid_size=3, spline_degree=2,
                       chunk_pixels=6, projector_chunk_size=7, **kwargs)
    bands.load_state_dict(full.state_dict())
    a = (torch.randn(1, 2, 8, 9) * 0.2).requires_grad_()
    b = a.detach().clone().requires_grad_()
    ya, yb = full(a), bands(b)
    torch.testing.assert_close(ya, yb, rtol=2e-5, atol=2e-6)
    ya.square().sum().backward()
    yb.square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(full.KAN_NM.spline_weight.grad,
                               bands.KAN_NM.spline_weight.grad, rtol=1e-4, atol=1e-5)


def test_linear_reference_matches_conv2d():
    layer = KAN_Conv2d(2, 3, kernel_size=3, padding=1,
                       projector_type='linear', chunk_pixels=4)
    x = torch.randn(1, 2, 7, 8)
    weight = layer.KAN_NM.core.weight.reshape(3, 2, 3, 3)
    expected = F.conv2d(x, weight, layer.KAN_NM.core.bias, padding=1)
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)


def test_update_grid_rejected_for_spline():
    layer = KAN_Conv2d(1, 1, grid_size=3)
    with pytest.raises(NotImplementedError):
        layer(torch.randn(1, 1, 3, 3), update_grid=True)
