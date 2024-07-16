import torch
import torchvision


class DummyDataset(torch.utils.data.Dataset):
    """Dummy torch datasets, includes joint data augmentation."""

    def __init__(
        self, input_data: torch.Tensor, label_data: torch.Tensor, size: int = 100
    ) -> None:
        self.input_data = input_data
        self.label_data = label_data
        self.size = size

        self.augmentor = torchvision.transforms.RandomRotation((-90, 90))

    def __len__(self):
        return self.size

    def __getitem__(self, _: int):
        state = (
            torch.get_rng_state()
        )  # Need to do this to ensure the rotations are equally applied to input:label.
        transformed_input = self.augmentor(self.input_data)
        torch.set_rng_state(state)
        transformed_label = self.augmentor(self.label_data)
        return transformed_input.repeat(3, 1, 1), transformed_label.repeat(3, 1, 1)
