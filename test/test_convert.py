import pytest
from torchvision.models.densenet import densenet121

import autonnx


@pytest.mark.parametrize("opset", [None, 14, 15, 16])
def test_convert(opset):
    model = densenet121()
    autonnx.convert(model)
