import argparse

import matplotlib.pyplot as plt
import numpy as np
import skimage


def _parse_args() -> dict:
    """Argument parser.

    :returns: dictionary of argument values.
    """
    parser = argparse.ArgumentParser(
        description="A script to generate some testing data for the conversion"
    )
    parser.add_argument(
        "--plot", "-p", action="store_true", help="whether or not to plot the images"
    )
    args = parser.parse_args()
    return vars(args)


def generate_images() -> "tuple[np.ndarray, np.ndarray]":
    """Generates test images from the camera image in scikit-image.

    :returns: pair of images of size 64x64 and type np.float32.
    """
    image = skimage.data.camera()[
        130:194, 240:304
    ]  # Select a crop, native size is 512, 512.
    image_noise = skimage.util.random_noise(image)

    image = image.astype(np.float32)
    image_noise = image_noise.astype(np.float32)

    return image, image_noise


def plot_images(image: np.ndarray, image_noise: np.ndarray) -> None:
    """Plots the pair of images.

    :param image: image (no noise) to plot, size = (sdim, sdim).
    :param image_noise: image (with noise) to plot, size = (sdim, sdim).
    """
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, cmap="gray")
    ax[0].title.set_text("Image")
    ax[1].imshow(image_noise, cmap="gray")
    ax[1].title.set_text("Image with noise")
    fig.tight_layout()
    plt.show()


def save_images(image: np.ndarray, image_noise: np.ndarray) -> None:
    """Saves the images to a PWD as png.

    :param image: image (no noise) to save, size = (sdim, sdim).
    :param image_noise: image (with noise) to save, size = (sdim, sdim).
    """
    plt.imsave("image.png", image)
    plt.imsave("image_noise.png", image_noise)
    print("Images saved.")


def main() -> None:
    args = _parse_args()
    image, image_noise = generate_images()
    if args["plot"]:
        plot_images(image, image_noise)
    save_images(image, image_noise)


if __name__ == "__main__":
    main()
