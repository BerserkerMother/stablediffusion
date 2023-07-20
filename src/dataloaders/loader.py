"""Contains code for dataloader"""
from torch.utils.data import DataLoader


class DataLoaderHelper:
    @staticmethod
    def create_loaders(dataset, config):
        train_loader = DataLoader(
            dataset=dataset["train"],
            batch_size=config.train_batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            dataset=dataset["test"],
            batch_size=config.eval_batch_size,
            shuffle=False
        )

        return train_loader, val_loader
