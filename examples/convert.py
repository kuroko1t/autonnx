import autonnx
import torchvision
from torchvision.models.densenet import densenet121

model = densenet121()
autonnx.convert(model, opset=16)
