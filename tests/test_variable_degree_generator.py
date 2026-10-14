"""Pruebas del perfil d_l, conectividad y compatibilidad de interfaz."""
import pytest
import torch

# Importar KAN_Conv2d registra nn.KAN_Conv2d, como en el proyecto original.
from KAN.KAN_Conv2d import KAN_Conv2d  # noqa: F401
from models.Generator import Generator
from models.ResidualBlock import ResidualBlock


def test_profile_assignment_and_external_degree():
    m = Generator(num_residual_blocks=3, scale_factor=2,
                  spline_degrees=[2, 3, 4], external_spline_degree=3,
                  grid_size=3, projector_type='spline')
    assert m.spline_degrees == (2, 3, 4)
    for block, degree in zip(m.res_blocks, (2, 3, 4)):
        assert block.spline_degree == degree
        for index in (0, 3):
            assert block.block[index].KAN_NM.basis.degree == degree
    assert m.initial[0].KAN_NM.basis.degree == 3
    assert m.post_res[0].KAN_NM.basis.degree == 3
    assert m.final[0].KAN_NM.basis.degree == 3


def test_uniform_and_default_profiles():
    assert Generator(num_residual_blocks=2, projector_type='linear',
                     spline_degrees=2).spline_degrees == (2, 2)
    assert Generator(num_residual_blocks=2, projector_type='linear').spline_degrees == (3, 3)


@pytest.mark.parametrize('kwargs', [
    {'num_residual_blocks': 3, 'spline_degrees': [2, 3]},
    {'num_residual_blocks': 1, 'spline_degrees': [0]},
    {'num_residual_blocks': 1, 'spline_degrees': '3'},
    {'num_residual_blocks': 0},
    {'num_residual_blocks': 1, 'external_spline_degree': 0},
])
def test_invalid_profiles(kwargs):
    with pytest.raises(ValueError):
        Generator(projector_type='linear', **kwargs)


def test_residual_block_shape_and_gradient():
    block = ResidualBlock(2, norm='none', spline_degree=2,
                          grid_size=3, projector_type='spline',
                          projector_chunk_size=8)
    x = torch.randn(1, 2, 3, 3, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert block.block[0].KAN_NM.spline_weight.grad is not None


def test_generator_forward_linear_small_image():
    model = Generator(num_residual_blocks=1, scale_factor=2,
                      norm='none', projector_type='linear',
                      spline_degrees=[2], chunk_pixels=32)
    with torch.no_grad():
        y = model(torch.randn(1, 3, 3, 3))
    assert y.shape == (1, 3, 6, 6)
    assert torch.isfinite(y).all()
