from torchvision.models.densenet import densenet121
import autonnx


def test_convert():
    model = densenet121()
    autonnx.convert(model)
