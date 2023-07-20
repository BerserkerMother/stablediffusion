""" Implements class for creating custom HF dataset """
from typing import List

from torchvision import transforms

import datasets


class FashionMnistDatasetBuilder:
    """Builder to build Mnist dataset"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = [] 

    def __repr__(self):
        print(self.dataset)

    def add_transforms(self, transform):
        if isinstance(transform, List):
            self.transform += transform
        else:
            self.transform.append(transform)
        return self

    def build(self):
        if len(self.transform) != 0:
            for t in self.transform:
                self.dataset.set_transform(t.get_transform_fn())
        return self.dataset

    @classmethod
    def intiliaze(cls):
        return cls(datasets.load_dataset("fashion_mnist"))


class Transforms:

    def __init__(self, transform):
        self.transform = transform

    @classmethod
    def from_config(cls, config):
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return cls(transform=transform)

    def get_transform_fn(self):
        def transform(examples):
            images = [self.transform(image)
                      for image in examples["image"]]
            return {"images": images}
        return transform
