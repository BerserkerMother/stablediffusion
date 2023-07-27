import os

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags


def _get_or_run(entrypoint, parameters, git_commit, use_cache: bool = True):
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, env_manager="local")
    return MlflowClient().get_run(submitted_run.run_id)


def workflow():
    with mlflow.start_run() as activate_run:
        git_commit = activate_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        load_train_run = _get_or_run(entrypoint="train", parameters={}, git_commit=git_commit)
        model_path = os.path.join(load_train_run.info.artifact_uri, "model/unet")
        print(model_path)
        train_train = _get_or_run(entrypoint="convert",
                                  parameters={"model_path": model_path,
                                              "convert_type": "onnx"
                                              },
                                  git_commit=git_commit)
        print("FINISHED!")
        print("Bet")


if __name__ == '__main__':
    workflow()
