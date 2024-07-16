import torch


class DummyModel(torch.nn.Module):
    """Dummy model to simply process an RGB image."""

    def __init__(self) -> None:
        super(DummyModel, self).__init__()
        self._initialise_architecture()

    def _initialise_architecture(self) -> None:  # Hardcoded as it's a dummy model.
        """Initialises the architecture of the model."""
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, padding="same"
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels=64, out_channels=3, kernel_size=3, padding="same"
            ),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        :param input_: The input image, expects shape [batch_size, 3, sdim, sdim]

        :returns: The output images, shape [batch_size, 3, sdim, sdim]
        """
        return self.network(input_)
