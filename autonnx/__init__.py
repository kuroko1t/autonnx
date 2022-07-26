import torch
import re
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("autonnx")


def convert(model, shape=[1, 3, 32, 32]):
    model_name = type(model).__name__.lower()
    dummy_input = torch.randn(*shape)
    for i in range(3):
        log.info(f"shape={shape}")
        try:
            torch.onnx.export(model, dummy_input, f"{model_name}.onnx")
            log.info(f"Convert to {model_name}.onnx")
            return True
        except Exception as e:
            error = str(e)
            m = re.search("to have (\d) channels", error)
            if m:
                expected_channel = int(m.group(1))
                shape[1] = expected_channel
            else:
                shape[2] *= 4
                shape[3] *= 4
                dummy_input = torch.randn(*shape)
    raise Exception(error)
