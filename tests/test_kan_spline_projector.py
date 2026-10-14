"""Pruebas del operador y = W_base SiLU(x) + sum C B(x) + b."""
import pytest
import torch
from KAN.KANSplineProjector import KANSplineProjector


@pytest.mark.parametrize('degree', [1, 2, 3, 4])
def test_shapes_and_finite(degree):
    model = KANSplineProjector(3, 5, grid_size=6, spline_degree=degree)
    assert model(torch.randn(7, 3)).shape == (7, 5)
    assert model(torch.randn(2, 3, 13)).shape == (2, 5, 13)
    assert torch.isfinite(model(torch.randn(7, 3))).all()
    assert model.spline_weight.shape == (5, 3, 6 + degree)


def test_gradients_reach_spline_coefficients_and_inputs():
    torch.manual_seed(1)
    model = KANSplineProjector(4, 3, chunk_size=5)
    x = (torch.rand(2, 4, 11) * 1.6 - 0.8).requires_grad_()
    model(x).square().mean().backward()
    assert model.spline_weight.grad is not None
    assert model.spline_weight.grad.abs().sum() > 0
    assert model.base_linear.weight.grad is not None
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_chunking_is_equivalent():
    torch.manual_seed(2)
    model = KANSplineProjector(3, 2, chunk_size=3)
    x = torch.rand(2, 3, 17) * 1.8 - 0.9
    y_chunked = model(x)
    model.chunk_size = 1000
    y_full = model(x)
    torch.testing.assert_close(y_chunked, y_full, atol=1e-6, rtol=1e-5)


def test_spline_branch_is_not_just_linear():
    model = KANSplineProjector(1, 1, grid_size=4, spline_degree=2, bias=False)
    with torch.no_grad():
        model.base_linear.weight.zero_()
        model.spline_weight.zero_()
        model.spline_weight[0, 0, 2] = 1.0
    x = torch.tensor([[-0.5], [0.0], [0.5]])
    y = model(x).squeeze()
    # Una función lineal tendría segunda diferencia exactamente cero.
    assert (y[0] - 2*y[1] + y[2]).abs() > 1e-4


def test_update_grid_explicitly_unsupported():
    model = KANSplineProjector(2, 3)
    with pytest.raises(NotImplementedError):
        model(torch.zeros(1, 2), update_grid=True)


def test_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip('CUDA no disponible')
    model = KANSplineProjector(3, 2).cuda()
    x = torch.rand(2, 3, 9, device='cuda', requires_grad=True)
    model(x).sum().backward()
    assert model.spline_weight.grad is not None
