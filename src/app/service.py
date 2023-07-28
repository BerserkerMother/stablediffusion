import onnxruntime as ort
import torch
import tqdm
from PIL import Image
from diffusers import DDIMScheduler
from flask import Flask, send_file
from tqdm import tqdm

from .flask import FlaskConfig


class OnnxModelWrapper:
    def __init__(self, config):
        self.scheduler = DDIMScheduler(num_train_timesteps=1000)
        self.model = ort.InferenceSession(config.model_path, providers=['CPUExecutionProvider'])

    def predict(self):

        # set step values
        self.scheduler.set_timesteps(500)
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
        grid.save("grid.jpg")
        return grid

    # convert_onnx()
    def make_grid(self, images, rows=4, cols=4):
        from PIL import Image

        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid


if __name__ == '__main__':
    flask_config = FlaskConfig()
    predictor = OnnxModelWrapper(flask_config.model_config)
    app = Flask("Stable Diffusion")


    @app.route("/infer", methods=["GET"])
    def infer():
        predictor.predict()
        return send_file("grid.jpg", mimetype="image/jpg")


    app.run(host=flask_config.host, port=flask_config.port)
