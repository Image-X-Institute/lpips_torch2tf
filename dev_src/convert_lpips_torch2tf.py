import argparse
import os

import lpips
import tensorflow as tf
import torch
import torchvision

from loss_fns import lpips_base_torch, lpips_base_tf
from utils import conversions, torch_tests, tf_tests


def _parse_args() -> dict:
    """Argument parser.

    :returns: dictionary of argument values.
    """
    parser = argparse.ArgumentParser(
        description="Convert an lpips torch loss function to onnx."
    )
    parser.add_argument(
        "--base",
        "-b",
        type=str,
        required=True,
        help="type of base network [options: 'alex', 'vgg16', 'squeeze', 'all']",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="whether or not to run comparison tests with PyPi model.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="whether or not to give verbose output.",
    )
    args = parser.parse_args()
    return vars(args)


def _run_torch_tests(loss_fn_base: torch.nn.Module, args: dict) -> None:
    """Testing the conversion from PyPi model to base torch model.

    :param loss_fn_base: the base torch model.
    :param args: command line arguments.
    """
    print("testing PyPi to base PyTorch conversion")
    image_torch = torchvision.io.read_image(
        os.path.relpath("data/image.png"), torchvision.io.ImageReadMode.GRAY
    ).float()

    image_noise_torch = torchvision.io.read_image(
        os.path.relpath("data/image_noise.png"), torchvision.io.ImageReadMode.GRAY
    ).float()

    if args["base"].lower() == "vgg16":
        loss_fn_pypi = lpips.LPIPS(net="vgg")
    else:
        loss_fn_pypi = lpips.LPIPS(net=args["base"])

    torch_tests.test_with_generated_data(
        loss_fn_pypi,
        loss_fn_base,
        image_torch,
        image_noise_torch,
    )

    torch_tests.test_with_random_data(
        loss_fn_pypi,
        loss_fn_base,
    )

    torch_tests.integration_test(loss_fn_base, image_torch, image_noise_torch)


def _run_tf_tests(loss_tf: tf.keras.Model) -> None:
    """Testing the tf model.

    :param loss_tf: the tf model.
    """
    image_torch = torchvision.io.read_image(
        os.path.relpath("data/image.png"), torchvision.io.ImageReadMode.GRAY
    ).float()

    image_noise_torch = torchvision.io.read_image(
        os.path.relpath("data/image_noise.png"), torchvision.io.ImageReadMode.GRAY
    ).float()

    tf_tests.integration_test(loss_tf, image_noise_torch, image_torch)


def _run_conversion_tests(loss_torch: torch.nn.Module, loss_tf: tf.keras.Model):
    """Testing the conversion from base torch model to tf model.

    :param loss_torch: the base torch model.
    :param loss_tf: the tf model.
    """
    print("testing pytorch to tensorflow conversion")
    image_np = torchvision.io.read_image(
        os.path.relpath("data/image.png"), torchvision.io.ImageReadMode.GRAY
    ).float()

    image_noise_np = torchvision.io.read_image(
        os.path.relpath("data/image_noise.png"), torchvision.io.ImageReadMode.GRAY
    ).float()

    tf_tests.test_with_random_data_base_net(loss_torch, loss_tf)
    tf_tests.test_with_generated_data(loss_torch, loss_tf, image_np, image_noise_np)


def _run_loading_tests(args: dict) -> None:
    """Testing the tf model when loading parameters from file.

    :param args: command line arguments.
    """
    print("testing the loading of tensorflow models from file")
    image_torch = torchvision.io.read_image(
        os.path.relpath("data/image.png"), torchvision.io.ImageReadMode.GRAY
    ).float()

    image_noise_torch = torchvision.io.read_image(
        os.path.relpath("data/image_noise.png"), torchvision.io.ImageReadMode.GRAY
    ).float()

    loss_tf_load = lpips_base_tf.LPIPS(
        args["base"],
    )

    tf_tests.integration_test(loss_tf_load, image_noise_torch, image_torch)


def _run(args: dict) -> None:
    """the main conversion function.

    :param args: command line arguments.
    """
    print(f"verbosity: {args['verbose']}")
    loss_fn_base = lpips_base_torch.LPIPS(base=args["base"]).eval()
    layers_tf = conversions.convert_loss_torch2tf(
        loss_fn_base, base_net=args["base"], verbose=args["verbose"]
    )
    loss_tf = lpips_base_tf.LPIPS(
        base=args["base"],
        base_weights=layers_tf["base_net_layers"],
        lin_weights=layers_tf["lin_layers"],
    )
    if args["test"]:
        _run_torch_tests(loss_fn_base, args)
        _run_conversion_tests(loss_fn_base, loss_tf)
        _run_tf_tests(loss_tf)

    _save_base_weights(loss_tf, args)
    _save_lins_weights(loss_tf, args)

    if args["test"]:
        _run_loading_tests(args)


def _save_base_weights(loss_tf: tf.keras.Model, args: dict) -> None:
    """Save the base net weights to file.

    :param loss_tf: the tf model.
    :param args: command line arguments.
    """
    os.makedirs(
        os.path.join("parameters", "lpips_tf", str(args["base"])), exist_ok=True
    )
    for key in list(loss_tf.base_net.network.keys()):
        loss_tf.base_net.network[key].save(
            os.path.join("parameters", "lpips_tf", args["base"], f"{key}_base.keras")
        )


def _save_lins_weights(loss_tf: tf.keras.Model, args: dict) -> None:
    """Save the lins net weights to file.

    :param loss_tf: the tf model.
    :param args: command line arguments.
    """
    os.makedirs(
        os.path.join("parameters", "lpips_tf", str(args["base"])), exist_ok=True
    )
    for i, lin in enumerate(loss_tf.lins):
        lin.save(os.path.join("parameters", "lpips_tf", args["base"], f"lin{i}.keras"))


def main() -> None:
    args = _parse_args()
    if args["base"].lower() == "all":
        for BASE in ["alex", "vgg16", "squeeze"]:
            args["base"] = BASE
            _run(args)
    else:
        _run(args)


if __name__ == "__main__":
    main()
