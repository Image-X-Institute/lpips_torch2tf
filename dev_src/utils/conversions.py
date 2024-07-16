import numpy as np

import keras
import tensorflow as tf
import torch
import torchvision


def convert_conv2d_torch2tf(
    torch_conv: torch.nn.modules.conv.Conv2d, verbose: bool = False, test: bool = True
) -> tf.keras.layers.Conv2D:
    """Converts a torch conv layer to a tf conv layer preserving weights and biases.

    :param torch_conv: the torch 2D convolutional layer.
    :param verbose: whether or not to provide verbose output.
    :param test: whether or not to test the conv layers on some dummy data.

    :returns: tensorflow 2D convolutional layer.
    """
    if verbose:
        print("converting torch conv2d layer to tensorflow conv2d layer")
        print("extracting conv weights")
    torch_weights = torch_conv.weight.detach().numpy()
    in_channels = torch_weights.shape[1]
    out_channels = torch_weights.shape[0]
    kernel_size_x = torch_weights.shape[2]
    kernel_size_y = torch_weights.shape[3]
    stride = torch_conv.stride
    padding = torch_conv.padding

    if padding not in ["valid", "same"]:
        create_pad_layer = True
    else:
        create_pad_layer = False

    if verbose:
        print("extracting conv bias, will skip if not detected")
    try:
        torch_bias = torch_conv.bias.detach().numpy()
        torch_bias_detected = True
    except AttributeError:
        torch_bias_detected = False
        if verbose:
            print("no kernel bias detected, skipping bias")

    if verbose:
        print("constructing tensorflow conv2d layer")
    input_shape = (1, kernel_size_x, kernel_size_y, in_channels)
    tf_conv = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=(kernel_size_x, kernel_size_y),
        use_bias=torch_bias_detected,
        strides=stride,
        padding=padding if padding in ["same", "valid"] else "valid",
    )
    _ = tf_conv(tf.random.normal(input_shape))

    if verbose:
        print("applying extracted weights and biases to tensorflow conv2d")
    if torch_bias_detected:
        tf_conv.set_weights(
            [np.swapaxes(np.swapaxes(torch_weights, 0, -1), 1, -2), torch_bias]
        )
    else:
        tf_conv.set_weights([np.swapaxes(np.swapaxes(torch_weights, 0, -1), 1, -2)])

    if create_pad_layer:
        pad = tf.keras.layers.ZeroPadding2D(padding)
        tf_conv = keras.Sequential([pad, tf_conv])

    if test and verbose:
        print("testing conversion")
        dummy_input = np.random.randn(
            1, in_channels, kernel_size_x + 11, kernel_size_y + 11
        ).astype(np.float32)

        yhat_torch = np.swapaxes(
            torch_conv(torch.from_numpy(dummy_input)).detach().numpy(), -1, 1
        )
        yhat_tf = tf_conv(tf.convert_to_tensor(np.swapaxes(dummy_input, -1, 1))).numpy()

        test_result = np.mean(np.abs(yhat_torch - yhat_tf)) / np.ptp(yhat_torch)
        print(f"conversion error (normalised (min-max) mean error): {test_result}")

    return tf_conv


def convert_ReLU_torch2tf(verbose: bool = False) -> tf.keras.layers.ReLU:
    """Generates a tf ReLU layer.

    :param verbose: whether or not to provide verbose output.

    :returns: tensorflow ReLU layer.
    """
    if verbose:
        print("converting torch ReLU layer to tensorflow ReLU layer")
    return tf.keras.layers.ReLU()


def convert_maxpool2d_torch2tf(
    torch_maxpool2d: torch.nn.modules.pooling.MaxPool2d, verbose: bool = False
) -> tf.keras.layers.MaxPool2D:
    """Converts a torch 2D max pooling layer to a tf 2D max pooling layer.

    :param torch_maxpool2d: the torch 2D max pooling layer.
    :param verbose: whether or not to provide verbose output.

    :returns: tensorflow 2D max pooling layer.
    """
    if verbose:
        print("converting torch maxpool2d layer to tensorflow maxpool2d layer")
    kernel_size = torch_maxpool2d.kernel_size
    stride = torch_maxpool2d.stride
    return tf.keras.layers.MaxPool2D(pool_size=kernel_size, strides=stride)


def convert_dropout_torch2tf(
    torch_dropout: torch.nn.modules.dropout.Dropout, verbose: bool = False
) -> tf.keras.layers.Dropout:
    """Converts a torch dropout layer to a tf dropout layer.

    :param torch_dropout: the torch dropout layer.
    :param verbose: whether or not to provide verbose output.

    :returns: tensorflow dropout layer.
    """
    if verbose:
        print("converting torch dropout layer to tensorflow dropout layer")
    return tf.keras.layers.Dropout(rate=torch_dropout.p)


def extract_base_net_layers_alex_vgg16_torch2tf(
    base_net_torch: torch.nn.Module, verbose: bool = False
) -> dict:
    """Converts the base net layers ('alex' or 'vgg16') from torch to tf.

    :param base_net_torch: the base network in torch.
    :param verbose: whether or not to provide verbose output.

    :returns: a dictionary containing each layer of the base net in tensorflow.
    """
    base_net_tf_layers = {}
    idx = 0
    for child in base_net_torch.named_children():
        for layer in child[1]:
            if isinstance(layer, str):
                for sublayer in base_net_torch.network[layer]:
                    match sublayer:
                        case torch.nn.modules.conv.Conv2d():
                            base_net_tf_layers[str(idx)] = convert_conv2d_torch2tf(
                                sublayer, verbose
                            )
                            idx += 1
                        case torch.nn.modules.activation.ReLU():
                            base_net_tf_layers[str(idx)] = convert_ReLU_torch2tf(
                                verbose
                            )
                            idx += 1
                        case torch.nn.modules.pooling.MaxPool2d():
                            base_net_tf_layers[str(idx)] = convert_maxpool2d_torch2tf(
                                sublayer, verbose
                            )
                            idx += 1
                        case _:
                            print(
                                f"layer type {type(layer)} not detected, skipping, be aware"
                            )
    return base_net_tf_layers


def extract_base_net_layers_squeeze_torch2tf(
    base_net_torch: torch.nn.Module, verbose: bool = False
) -> dict:
    """Converts the base net layers ('squeeze') from torch to tf.

    :param base_net_torch: the base network in torch.
    :param verbose: whether or not to provide verbose output.

    :returns: a dictionary containing each layer of the base net in tensorflow.
    """
    base_net_tf_layers = {}
    idx = 0
    for child in base_net_torch.named_children():
        for layer in child[1]:
            match layer:
                case torch.nn.modules.conv.Conv2d():
                    base_net_tf_layers[str(idx)] = convert_conv2d_torch2tf(
                        layer, verbose
                    )
                    idx += 1
                case torch.nn.modules.activation.ReLU():
                    base_net_tf_layers[str(idx)] = convert_ReLU_torch2tf(verbose)
                    idx += 1
                case torch.nn.modules.pooling.MaxPool2d():
                    base_net_tf_layers[str(idx)] = convert_maxpool2d_torch2tf(
                        layer, verbose
                    )
                    idx += 1
                case torchvision.models.squeezenet.Fire():  # Squeezenet specific.
                    for grandchild in layer.named_children():
                        match grandchild[1]:
                            case torch.nn.modules.conv.Conv2d():
                                base_net_tf_layers[str(idx)] = convert_conv2d_torch2tf(
                                    grandchild[1], verbose
                                )
                                idx += 1
                            case torch.nn.modules.activation.ReLU():
                                base_net_tf_layers[str(idx)] = convert_ReLU_torch2tf(
                                    verbose
                                )
                                idx += 1
                            case torch.nn.modules.pooling.MaxPool2d():
                                base_net_tf_layers[str(idx)] = (
                                    convert_maxpool2d_torch2tf(grandchild[1], verbose)
                                )
                                idx += 1
                            case _:
                                print(
                                    f"layer type {type(layer)} not detected, skipping, be aware"
                                )
                case _:
                    print(f"layer type {type(layer)} not detected, skipping, be aware")
        return base_net_tf_layers


def extract_base_net_layers_torch2tf(
    base_net_torch: torch.nn.Module, model_name: str = "alex", verbose: bool = False
) -> dict:
    """Extracts the layers from the PyTorch base net and converts them to Tensorflow.

    :param base_net_torch: the base net layers in pytorch.
    :param model_name: the base net model name.
    :param test: whether or not to run a conversion test.

    :returns: a dictionary of tensorflow layers that is understandable by the LPIPS tf class.
    """
    base_net_tf_layers = {}
    match model_name.lower():
        case "alex" | "vgg16":
            print(f"extracting parameters from base net: {model_name}")
            base_net_tf_layers = extract_base_net_layers_alex_vgg16_torch2tf(
                base_net_torch, verbose
            )

        case "squeeze":
            print(f"extracting parameters from base net: {model_name}")
            base_net_tf_layers = extract_base_net_layers_squeeze_torch2tf(
                base_net_torch, verbose
            )

        case _:
            raise ValueError(
                "Base net not supported, options are ['alex', 'vgg16', 'squeeze']"
            )

    return base_net_tf_layers


def extract_lin_layers_torch2tf(
    lin_torch: torch.nn.Module, verbose: bool = False
) -> dict:
    """Extracts the layers from the PyTorch lin net and converts them to Tensorflow.

    :param lin_torch: the base net layers in pytorch.
    :param test: whether or not to run a conversion test.

    :returns: a dictionary of tensorflow layers that is understandable by the LPIPS tf class.
    """
    lin_tf_layers = {}
    print("extracting lin layers")
    idx = 0
    for child in lin_torch.named_children():
        for grandchild in child[1].named_children():
            for layer in grandchild[1].named_children():
                match layer[1]:
                    case torch.nn.modules.conv.Conv2d():
                        lin_tf_layers[str(idx)] = convert_conv2d_torch2tf(
                            layer[1], verbose
                        )
                        idx += 1
                    case torch.nn.modules.activation.ReLU():
                        lin_tf_layers[str(idx)] = convert_ReLU_torch2tf(verbose)
                        idx += 1
                    case torch.nn.modules.pooling.MaxPool2d():
                        lin_tf_layers[str(idx)] = convert_maxpool2d_torch2tf(
                            layer[1], verbose
                        )
                        idx += 1
                    case torch.nn.modules.dropout.Dropout():
                        lin_tf_layers[str(idx)] = convert_dropout_torch2tf(
                            layer[1], verbose
                        )
                        idx += 1
                    case _:
                        print(
                            f"layer type {type(layer[1])} not detected, skipping, adding None in case we dont want dropout, be aware"
                        )
                        lin_tf_layers[str(idx)] = None
                        idx += 1
    return lin_tf_layers


def convert_loss_torch2tf(
    loss_torch: torch.nn.Module, base_net: str = "alex", verbose: bool = False
) -> dict:
    layers_tf = dict()

    for child in loss_torch.named_children():
        if child[0] == "base_net":
            layers_tf["base_net_layers"] = extract_base_net_layers_torch2tf(
                child[1], base_net, verbose
            )
        elif child[0] == "lins":
            layers_tf["lin_layers"] = extract_lin_layers_torch2tf(child[1], verbose)
        else:
            pass
    return layers_tf
