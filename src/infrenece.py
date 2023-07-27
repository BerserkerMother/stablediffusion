from sys import version_info

import PIL
import diffusers
import mlflow
import onnxruntime
import torch
import tqdm

PYTHON_VERSION = "{major}.{minor}.{micro}".format(
    major=version_info.major, minor=version_info.minor, micro=version_info.micro
)


class OnnxModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from diffusers import DDIMScheduler
        import onnxruntime as ort

        self.scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.model = ort.InferenceSession(context.artifacts["model"])

    def predict(self, context, model_input):
        from PIL import Image
        from tqdm import tqdm
        import torch

        # set step values
        self.scheduler.set_timesteps(1000)
        image = torch.randn(16, 1, 32, 32, dtype=torch.float32)

        for t in tqdm(self.scheduler.timesteps):
            # 1. predict noise model_output
            image = image.numpy()
            t = t.numpy().reshape(-1)
            model_in = {
                "sample": image,
                "timestep": t
            }
            model_output = self.model.run(None, model_in)[0]

            # 2. compute previous image: x_t -> x_t-1
            image = torch.tensor(image)
            model_output = torch.tensor(model_output)
            t = torch.tensor(t)
            image = self.scheduler.step(
                model_output, t, image).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        images = [Image.fromarray(image.reshape(32, 32), mode="L") for image in images]
        grid = self.make_grid(images)
        grid.save("test1.jpg")

    # convert_onnx()
    def make_grid(self, images, rows=4, cols=4):
        from PIL import Image

        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid


# convert_onnx()
def prep_serving(model_location="model.onnx", output_path="serving_model"):
    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            "python={}".format(PYTHON_VERSION),
            "pip",
            {
                "pip": [
                    "mlflow=={}".format(mlflow.__version__),
                    "torch=={}".format(torch.__version__),
                    "PIL=={}".format(PIL.__version__),
                    "onnxruntime=={}".format(onnxruntime.__version__),
                    "tqdm=={}".format(tqdm.__version__),
                    "diffusers=={}".format(diffusers.__version__),
                ],
            },
        ],
        "name": "stable_diffusion",
    }
    artifacts = {"model": model_location}
    mlflow.pyfunc.save_model(
        path=output_path,
        python_model=OnnxModelWrapper(),
        artifacts=artifacts,
        conda_env=conda_env,
    )


if __name__ == '__main__':
    prep_serving()
