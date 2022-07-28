# autonnx

Exports a torch model to an onnx model without specifying the input shape.

## Example

```python3
import autonnx
import torchvision
from torchvision.models.densenet import densenet121

model = densenet121()
autonnx.convert(model)
```