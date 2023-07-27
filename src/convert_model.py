import os
import shutil

import click
import mlflow
import torch
from diffusers import UNet2DModel
from onnxruntime.quantization import quantize_dynamic, QuantType

from infrenece import prep_serving


def convert_to_onnx(model_path: str, quantize: bool = False):
    model = UNet2DModel.from_pretrained(model_path)
    model.eval()
    timesteps = torch.randint(0, 4, (10,), ).long()
    dummy_model_input = torch.randn(10, 1, 32, 32), timesteps
    if not os.path.exists("model"):
        os.mkdir("model")
    save_path = "model/model.onnx"
    torch.onnx.export(
        model,
        dummy_model_input,
        f="model/onnx.model",
        input_names=['sample', 'timestep'],
        output_names=['out'],
        dynamic_axes={'sample': {0: 'batch_size'},
                      'out': {0: "batch_size"},
                      "timestep": {0: "batch_size"}
                      },
        do_constant_folding=True,
        opset_version=13,
    )
    if quantize:
        path = save_path.split(".")[:-1] + ["quantize", "onnx"]
        path = ".".join(path)
        _ = quantize_dynamic(save_path, path, weight_type=QuantType.QUInt8)

    with mlflow.start_run() as active_run:
        mlflow.log_artifact("model", "model")
        shutil.rmtree("model")
        return active_run.info.artifact_uri


@click.command()
@click.option("--convert_type", help="type to convert to")
@click.option("--model_path")
@click.option("--quantize", default=True)
def main(convert_type, model_path, quantize):
    if convert_type == "onnx":
        artifact_uri = convert_to_onnx(model_path)
    else:
        raise Exception("Not implements!")

    # setup custom model for serving
    path = os.path.join(artifact_uri, "model/model.onnx")
    prep_serving(path, ".")


if __name__ == '__main__':
    main()
