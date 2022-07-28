import logging
import os
import re

import onnx
import torch
from onnxsim import simplify
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

log = logging.getLogger("autonnx")


def convert(model, opset=None, shape=[1, 3, 32, 32]):
    model_name = type(model).__name__.lower()
    dummy_input = torch.randn(*shape)
    success_flag = False
    for i in range(5):
        log.info(f"shape={shape}")
        try:
            torch.onnx.export(
                model, dummy_input, f"{model_name}_origin.onnx", opset_version=opset
            )
            log.info(f"torch.onnx.export is success")
            success_flag = True
            break
        except Exception as e:
            error = str(e)
            m = re.search("to have (\d) channels", error)
            if m:
                expected_channel = int(m.group(1))
                shape[1] = expected_channel
            else:
                shape[2] *= 2
                shape[3] *= 2
                dummy_input = torch.randn(*shape)
    if not success_flag:
        raise Exception(error)
    model = onnx.load(f"{model_name}_origin.onnx")
    # convert model
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, f"{model_name}.onnx")
    log.info(f"export to {model_name}.onnx")
    os.remove(f"{model_name}_origin.onnx")
