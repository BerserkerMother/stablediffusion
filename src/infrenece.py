import torch
import numpy as np

from diffusers import UNet2DModel, DDIMPipeline, DDIMScheduler
from optimum.onnxruntime import ORTStableDiffusionPipeline
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from PIL import Image
from tqdm import tqdm



def convert_onnx():
    model = UNet2DModel.from_pretrained(
        "/home/kave/ml_journey/Stable Diffusion/ddpm-fashion_mnist/unet")
    model.eval()
    timesteps = torch.randint(0, 4, (10,),).long()
    dummy_model_input = torch.randn(10, 1, 32, 32)
    torch.onnx.export(
        model,
        (dummy_model_input, timesteps),
        f="diffuser.onnx",
        input_names=['sample', 'timestep'],
        output_names=['out'],
        dynamic_axes={'sample': {0: 'batch_size'},
                      'out': {0: "batch_size"},
                      "timestep": {0: "batch_size"}
                      },
        do_constant_folding=True,
        opset_version=13,
    )
    quantized_model = quantize_dynamic("diffuser.onnx", "diffuser.quantize.onnx", weight_type=QuantType.QUInt8)


def infer():
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    ort_session = ort.InferenceSession("diffuser.quantize.onnx")
    # set step values
    scheduler.set_timesteps(1000)
    image = torch.randn(16, 1, 32, 32, dtype=torch.float32)
    
    for t in tqdm(scheduler.timesteps):
        # 1. predict noise model_output
        image = image.numpy()
        t = t.numpy().reshape(-1)
        model_in = {
            "sample": image,
            "timestep": t
        }
        model_output = ort_session.run(None, model_in)[0]

        # 2. compute previous image: x_t -> x_t-1
        image = torch.tensor(image)
        model_output = torch.tensor(model_output)
        t = torch.tensor(t)
        image = scheduler.step(
            model_output, t, image).prev_sample

    print(image.shape)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    images = [Image.fromarray(image.reshape(32, 32), mode="L") for image in images]
    grid = make_grid(images)
    grid.save("test1.jpg")

# convert_onnx()
def make_grid(images, rows=4, cols=4):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

# convert_onnx()
infer()