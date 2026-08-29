import numpy as np

from navigen_perception.backends import base
from navigen_perception.backends.mock_backend import MockSegmentationBackend


def test_mock_backend_contract():
    backend = MockSegmentationBackend()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    result = backend.infer(image)
    assert isinstance(result, base.SegmentationResult)
    assert result.mask.shape == (240, 320)
    assert result.mask.dtype == np.uint8
    assert 0.0 <= result.confidence <= 1.0
    assert set(np.unique(result.mask)).issubset(
        {base.UNKNOWN, base.TRAVERSABLE, base.NON_TRAVERSABLE, base.OBSTACLE})


def test_mock_backend_ground_is_traversable():
    backend = MockSegmentationBackend()
    result = backend.infer(np.zeros((240, 320, 3), dtype=np.uint8))
    # bottom-center in front of the robot must be traversable
    assert result.mask[230, 160] == base.TRAVERSABLE
    # top region (sky/far field) must not be traversable
    assert result.mask[10, 160] == base.NON_TRAVERSABLE


def test_colorize_shapes_and_colors():
    mask = np.array([[base.UNKNOWN, base.TRAVERSABLE],
                     [base.NON_TRAVERSABLE, base.OBSTACLE]], dtype=np.uint8)
    color = base.colorize(mask)
    assert color.shape == (2, 2, 3)
    assert tuple(color[0, 1]) == base.CLASS_COLORS_BGR[base.TRAVERSABLE]
    assert tuple(color[1, 1]) == base.CLASS_COLORS_BGR[base.OBSTACLE]
