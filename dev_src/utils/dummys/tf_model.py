import tensorflow as tf


class DummyModel(tf.keras.Model):
    """Dummy model to simple process an RGB image."""

    def __init__(self) -> None:
        super(DummyModel, self).__init__()
        self._initialise_architecture()

    def _initialise_architecture(self) -> None:  # Hardcoded as it's a dummy model.
        """Initialises the architecture of the model."""
        self.network = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(None, None, 3)),  # RGB
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, activation="relu", padding="same"
                ),
                tf.keras.layers.Conv2D(
                    filters=3, kernel_size=3, activation="relu", padding="same"
                ),
            ]
        )
