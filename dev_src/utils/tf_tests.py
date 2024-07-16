import numpy as np
import tensorflow as tf
import torch

from utils.dummys import tf_model


def test_with_generated_data(
    loss_torch: torch.nn.Module,
    loss_tf: tf.keras.Model,
    image_0: torch.Tensor,
    image_1: torch.Tensor,
) -> None:
    """Tests whether the loss function gives the correct output based on pre-generated data.

    :param loss_torch: the pytorch loss function.
    :param loss_tf: the tensorflow loss function.
    :param image_0: an image to test (expects it to be grayscale [1, sdim, sdim]).
    :param image_1: an image to test (expects it to be grayscale [1, sdim, sdim]).
    """
    print("running tests on generated data...")

    image_0_tf = tf.expand_dims(
        tf.transpose(tf.repeat(tf.convert_to_tensor(image_0), 3, 0), [2, 1, 0]), 0
    )
    image_1_tf = tf.expand_dims(
        tf.transpose(tf.repeat(tf.convert_to_tensor(image_1), 3, 0), [2, 1, 0]), 0
    )
    loss_torch_output = loss_torch(
        image_0.repeat(3, 1, 1).unsqueeze(0), image_1.repeat(3, 1, 1).unsqueeze(0)
    )
    loss_tf_output = loss_tf(image_0_tf, image_1_tf, training=False)

    print(f"torch loss output: {loss_torch_output}")
    print(f"tensorflow loss output: {loss_tf_output}")

    print(f"error: {np.abs(loss_torch_output.detach().numpy()-loss_tf_output.numpy())}")
    print(88 * "-")


def test_with_random_data_base_net(
    loss_torch: torch.nn.Module, loss_tf: tf.keras.Model, num_test: int = 100
):
    """Tests the conversion error based on randomly generated data.

    :param loss_torch: the pytorch loss function.
    :param loss_tf: the tensorflow loss function.
    :param num_test: The number of random tests to run.
    """
    print("running tests with random data")
    err = []
    for _ in range(num_test):
        dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)

        yhat_torch = loss_torch.base_net(torch.from_numpy(dummy_input))
        yhat_tf = loss_tf.base_net(
            tf.convert_to_tensor(np.swapaxes(dummy_input, -1, 1)), training=False
        )

        err_total_mini = 0.0
        for i in range(len(yhat_torch)):
            yhat_torch_out = np.swapaxes(yhat_torch[i].detach().numpy(), -1, 1)
            yhat_tf_out = yhat_tf[i].numpy()
            test_result = np.mean(np.abs(yhat_torch_out - yhat_tf_out)) / np.ptp(
                yhat_torch_out
            )
            err_total_mini += test_result
        err.append(np.mean(err_total_mini))
    print(f"conversion error (mean error): {np.mean(err)}")


def integration_test(
    loss_tf: tf.keras.Model, image_0: torch.Tensor, image_1: torch.Tensor
) -> None:
    """Integration test showing the loss can work in a training loop.

    :param loss_fn_base: the loss function that was re-created in this repository.
    :param image_0: a label image to test (expects it to be grayscale [1, sdim, sdim]).
    :param image_1: an input image to test (expects it to be grayscale [1, sdim, sdim]).
    """
    print("commencing integration test")
    image_noise_tf = tf.expand_dims(
        tf.transpose(tf.repeat(tf.convert_to_tensor(image_0.numpy()), 3, 0), [2, 1, 0]),
        0,
    )

    image_tf = tf.expand_dims(
        tf.transpose(tf.repeat(tf.convert_to_tensor(image_1.numpy()), 3, 0), [2, 1, 0]),
        0,
    )

    model_tf = tf_model.DummyModel()

    model_tf.network.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_tf,
        metrics=["mean_squared_error"],
    )

    model_tf.network.summary()

    _ = model_tf.network.fit(image_noise_tf, image_tf, epochs=30)
