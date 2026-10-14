"""Pruebas de propiedades matemáticas de las bases B-spline."""

import pytest
import torch

from KAN.BSplineBasis import BSplineBasis


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_partition_unity_and_nonnegative(degree):
    """En [a,b], B_i,d >= 0 y sum_i B_i,d = 1."""
    layer = BSplineBasis(grid_size=8, degree=degree)
    x = torch.linspace(-1, 1, 201, dtype=torch.float64)
    y = layer(x)
    assert y.shape == (201, 8 + degree)
    torch.testing.assert_close(y.sum(-1), torch.ones_like(x), atol=1e-10, rtol=1e-10)
    assert bool((y >= -1e-12).all())


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_local_support(degree):
    """B_i,d(x) = 0 fuera de [t_i, t_{i+d+1}]."""
    layer = BSplineBasis(grid_size=8, degree=degree)
    x = torch.linspace(-1.5, 1.5, 301, dtype=torch.float64)
    y = layer(x)
    t = layer.knots
    for i in range(layer.n_basis):
        outside = (x < t[i]) | (x > t[i + degree + 1])
        assert bool((y[outside, i] == 0).all())


def test_gradients_inside_domain():
    """Autograd debe propagar gradientes de las bases a x."""
    layer = BSplineBasis(grid_size=8, degree=3)
    x = torch.tensor([-0.71, -0.13, 0.37], requires_grad=True)
    weights = torch.arange(layer.n_basis, dtype=torch.float32)
    loss = (layer(x) * weights).sum()
    grad = torch.autograd.grad(loss, x)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_outside_domain_zero():
    layer = BSplineBasis(grid_size=8, degree=3)
    y = layer(torch.tensor([-2.0, 2.0]))
    assert bool((y == 0).all())


def test_cpu_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA no disponible")
    layer = BSplineBasis(grid_size=8, degree=3).cuda()
    x = torch.linspace(-1, 1, 32, device="cuda")
    y = layer(x)
    assert y.is_cuda
    torch.testing.assert_close(y.sum(-1), torch.ones_like(x), atol=1e-5, rtol=1e-5)
