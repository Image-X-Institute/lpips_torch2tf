from collections import namedtuple
import os

import tensorflow as tf
import keras


class LPIPS(tf.keras.Model):
    def __init__(
        self,
        base: str = "alex",
        base_weights: dict = None,
        lin_weights: dict = None,
        spatial: bool = True,
        pre_norm: bool = False
    ):
        super(LPIPS, self).__init__()
        """The LPIPS loss function.
        
        :param base: the base model, options are 'alex', 'vgg16', or 'squeeze'.
        :param base_weights: the base model weights from the torch2tf conversion.
        :param lin_weights: the lin model weights from the torch2tf conversion.
        :param spatial: add spatial averaging?
        :param pre_norm: pre-normalise the input data between [-1,1] using min-max scaling?
        """
        self.base = base
        self.base_weights = base_weights

        self.lin_weights = lin_weights

        self.spatial = spatial
        self.pre_norm = pre_norm

        self.scaling_layer = _ScalingLayer()

        self.parameter_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "parameters", "lpips_tf"
        )

        self._parse_base()

        if lin_weights is None:
            self._load_architecture_from_file()
        else:
            self._initialise_lpips()

    def _parse_base(self):
        print(f"fetching base: {self.base}")
        match self.base.lower():
            case "alex":
                self.base_net = _AlexNet(self.parameter_path, self.base_weights)
                self.chs = [64, 192, 384, 256, 256]
                self.model_path_name = "alex"
            case "vgg16":
                self.base_net = _VGG16Net(self.parameter_path, self.base_weights)
                self.chs = [64, 128, 256, 512, 512]
                self.model_path_name = "vgg"
            case "squeeze":
                self.base_net = _SqueezeNet(self.parameter_path, self.base_weights)
                self.chs = [64, 128, 256, 384, 384, 512, 512]
                self.model_path_name = "squeeze"
            case _:
                raise ValueError(
                    "Base net not supported, options are ['alex', 'vgg16', 'squeeze']"
                )

        self.L = len(self.chs)

    def _load_architecture_from_file(self):
        print("loading lins parameters from file")
        match self.base.lower():
            case "alex":
                self.lin0 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "alex", "lin0.keras")
                )
                self.lin1 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "alex", "lin1.keras")
                )
                self.lin2 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "alex", "lin2.keras")
                )
                self.lin3 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "alex", "lin3.keras")
                )
                self.lin4 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "alex", "lin4.keras")
                )

                self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

            case "vgg16":
                self.lin0 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "vgg16", "lin0.keras")
                )
                self.lin1 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "vgg16", "lin1.keras")
                )
                self.lin2 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "vgg16", "lin2.keras")
                )
                self.lin3 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "vgg16", "lin3.keras")
                )
                self.lin4 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "vgg16", "lin4.keras")
                )

                self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

            case "squeeze":
                self.lin0 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "squeeze", "lin0.keras")
                )
                self.lin1 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "squeeze", "lin1.keras")
                )
                self.lin2 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "squeeze", "lin2.keras")
                )
                self.lin3 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "squeeze", "lin3.keras")
                )
                self.lin4 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "squeeze", "lin4.keras")
                )
                self.lin5 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "squeeze", "lin5.keras")
                )
                self.lin6 = tf.keras.models.load_model(
                    os.path.join(self.parameter_path, "squeeze", "lin6.keras")
                )

                self.lins = [
                    self.lin0,
                    self.lin1,
                    self.lin2,
                    self.lin3,
                    self.lin4,
                    self.lin5,
                    self.lin6,
                ]

            case _:
                raise ValueError(
                    "Base net not supported, options are ['alex', 'vgg16', 'squeeze']"
                )

    def _initialise_lpips(self):
        match self.base.lower():
            case "alex" | "vgg16":
                self.lin0 = _NetLinLayer(self.lin_weights["1"], self.lin_weights["0"])
                self.lin1 = _NetLinLayer(self.lin_weights["3"], self.lin_weights["2"])
                self.lin2 = _NetLinLayer(self.lin_weights["5"], self.lin_weights["4"])
                self.lin3 = _NetLinLayer(self.lin_weights["7"], self.lin_weights["6"])
                self.lin4 = _NetLinLayer(self.lin_weights["9"], self.lin_weights["8"])

                self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

            case "squeeze":
                self.lin0 = _NetLinLayer(self.lin_weights["1"], self.lin_weights["0"])
                self.lin1 = _NetLinLayer(self.lin_weights["3"], self.lin_weights["2"])
                self.lin2 = _NetLinLayer(self.lin_weights["5"], self.lin_weights["4"])
                self.lin3 = _NetLinLayer(self.lin_weights["7"], self.lin_weights["6"])
                self.lin4 = _NetLinLayer(self.lin_weights["9"], self.lin_weights["8"])
                self.lin5 = _NetLinLayer(self.lin_weights["11"], self.lin_weights["10"])
                self.lin6 = _NetLinLayer(self.lin_weights["13"], self.lin_weights["12"])

                self.lins = [
                    self.lin0,
                    self.lin1,
                    self.lin2,
                    self.lin3,
                    self.lin4,
                    self.lin5,
                    self.lin6,
                ]

            case _:
                raise ValueError(
                    "Base net not supported, options are ['alex', 'vgg16', 'squeeze']"
                )

    def _normalise(self, input_: tf.Tensor, type: str = "pre") -> tf.Tensor:
        """Normalises the input tensor, has two modes of operation.

        :param input_: the input tensor to be normalised.
        :param type: the type of normalisation based [options: 'pre' and 'post'].

        :returns: the normalised tensor.
        """
        match type.lower():
            case "pre":
                return 2 * input_ - 1
            case "post":
                return input_ / (
                    tf.sqrt(tf.math.reduce_sum(input_**2, axis=-1, keepdims=True))
                    + 1e-10
                )

    def _upsample(
        self, input_: tf.Tensor, out_HW: tuple = (64, 64), mode: str = "bilinear"
    ) -> tf.Tensor:
        """Resizes the input tensor (image).

        :param input_: the input image to be resized.
        :param out_HW: the desired height/width of the returned image.
        :param mode: the interpolation mode.

        :returns: the resized image.
        """

        return tf.image.resize(input_, size=out_HW, method=mode)

    def _spatial_average(self, input_: tf.Tensor, keepdim: bool = True) -> tf.Tensor:
        """Performs a spatial average over the last two dims.

        :param input_: the input image to be averaged.
        :param keepdim: whether or not to compress the out tensor.

        :returns: the spatially averaged image.
        """
        return tf.math.reduce_mean(input_, [1, 2], keepdims=keepdim)

    def _generate_features_and_differences(
        self, output_0: tf.Tensor, output_1: tf.Tensor
    ) -> dict:
        """Generates the features and differences from the base network.

        :param output_0: the image_0 output of the base net.
        :param output_1: the image_1 output of the base net.

        :returns: the diffs of the individual layers.
        """
        features_0, features_1, diffs = {}, {}, {}

        for kk in range(self.L):
            features_0[kk], features_1[kk] = self._normalise(
                output_0[kk], "post"
            ), self._normalise(output_1[kk], "post")
            diffs[kk] = (features_0[kk] - features_1[kk]) ** 2

        return diffs
    
    def _pre_normalise(self, input_: tf.Tensor) -> tf.Tensor:
        """Pre-normalises an input tensor between [-1,1] using min-max scaling.
        
        :param input_: the input tensor.

        :returns: the normalised [-1,1] tensor.
        """
        min_val = tf.math.reduce_min(input_)
        max_val = tf.math.reduce_max(input_)
        return (2 * (input_ - min_val) / (max_val - min_val)) - 1

    def call(
        self,
        image_0: tf.Tensor,
        image_1: tf.Tensor,
        retPerLayer: bool = False,
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
        """Forward function of the loss calculation.

        :param image_0: the reference image.
        :param image_1: the comparator image.
        :param retPerLayer: whether to return the residual per layer with the output.
        """
        if self.pre_norm:
            image_0, image_1 = self._pre_normalise(image_0), self._pre_normalise(image_1)

        image_0_scaled, image_1_scaled = self.scaling_layer(
            image_0
        ), self.scaling_layer(image_1)

        output_0, output_1 = self.base_net(image_0_scaled), self.base_net(
            image_1_scaled
        )

        diffs = self._generate_features_and_differences(output_0, output_1)

        if not self.spatial:
            res = [
                self._upsample(self.lins[kk](diffs[kk]), out_HW=image_0.shape[-2:])
                for kk in range(self.L)
            ]
        else:
            res = [
                self._spatial_average(self.lins[kk](diffs[kk])) for kk in range(self.L)
            ]

        val = 0
        for l in range(self.L):
            val += res[l]

        if retPerLayer:
            return (val, res)

        else:
            return val


class _ScalingLayer(tf.keras.Model):
    def __init__(self) -> None:
        super(_ScalingLayer, self).__init__()
        self.shift = tf.Variable(
            initial_value=[-0.030, -0.088, -0.188], trainable=False
        )[None, None, None, :]
        self.scale = tf.Variable(initial_value=[0.458, 0.448, 0.450], trainable=False)[
            None, None, None, :
        ]

    def call(self, input_: tf.Tensor) -> tf.Tensor:
        """Implements a scaling for the input.

        :param input_: the input tensor to be scaled.

        :returns: the scaled tensor.
        """
        return (input_ - self.shift) / self.scale


class _AlexNet(tf.keras.Model):
    def __init__(self, parameter_path: str, weights_: dict = None) -> None:
        super(_AlexNet, self).__init__()

        self.parameter_path = os.path.join(parameter_path, "alex")

        if weights_ is None:
            self._load_architecture_from_file()
        else:
            self.weights_ = weights_
            self._initialise_architecture()

        self.n_slices = 5

    def _load_architecture_from_file(self):
        print("loading base parameters from file")
        self.network = {
            "slice_1": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_1_base.keras")
            ),
            "slice_2": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_2_base.keras")
            ),
            "slice_3": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_3_base.keras")
            ),
            "slice_4": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_4_base.keras")
            ),
            "slice_5": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_5_base.keras")
            ),
        }

    def _initialise_architecture(self):
        self.network = {
            "slice_1": tf.keras.Sequential(
                [
                    tf.keras.Input(
                        shape=(None, None, None)
                    ),  # something we have to do with tf...
                    self.weights_["0"],
                    self.weights_["1"],
                ]
            ),
            "slice_2": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    self.weights_["2"],
                    self.weights_["3"],
                    self.weights_["4"],
                ]
            ),
            "slice_3": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    self.weights_["5"],
                    self.weights_["6"],
                    self.weights_["7"],
                ]
            ),
            "slice_4": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    self.weights_["8"],
                    self.weights_["9"],
                ]
            ),
            "slice_5": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    self.weights_["10"],
                    self.weights_["11"],
                ]
            ),
        }

    def call(self, input_: tf.Tensor) -> tf.Tensor:
        """Forward pass of the network.

        :param input_: the input image, expects shape [batch_size, sdim, sdim, 3]

        :returns: the output containing the intermediate activations.
        """
        h = self.network["slice_1"](input_)
        h_relu1 = h

        h = self.network["slice_2"](h)
        h_relu2 = h

        h = self.network["slice_3"](h)
        h_relu3 = h

        h = self.network["slice_4"](h)
        h_relu4 = h

        h = self.network["slice_5"](h)
        h_relu5 = h

        outputs = namedtuple(
            "AlexOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5"]
        )
        return outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


class _VGG16Net(tf.keras.Model):
    def __init__(self, parameter_path: str, weights_: dict = None) -> None:
        super(_VGG16Net, self).__init__()

        self.parameter_path = os.path.join(parameter_path, "vgg16")

        if weights_ is None:
            self._load_architecture_from_file()
        else:
            self.weights_ = weights_
            self._initialise_architecture()

        self.n_slices = 5

    def _load_architecture_from_file(self):
        print("loading base parameters from file")
        self.network = {
            "slice_1": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_1_base.keras")
            ),
            "slice_2": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_2_base.keras")
            ),
            "slice_3": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_3_base.keras")
            ),
            "slice_4": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_4_base.keras")
            ),
            "slice_5": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_5_base.keras")
            ),
        }

    def _initialise_architecture(self):
        self.network = {
            "slice_1": tf.keras.Sequential(
                [
                    tf.keras.Input(
                        shape=(None, None, None)
                    ),  # something we have to do with tf...
                    self.weights_["0"],
                    self.weights_["1"],
                    self.weights_["2"],
                    self.weights_["3"],
                ]
            ),
            "slice_2": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    self.weights_["4"],
                    self.weights_["5"],
                    self.weights_["6"],
                    self.weights_["7"],
                    self.weights_["8"],
                ]
            ),
            "slice_3": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    self.weights_["9"],
                    self.weights_["10"],
                    self.weights_["11"],
                    self.weights_["12"],
                    self.weights_["13"],
                    self.weights_["14"],
                    self.weights_["15"],
                ]
            ),
            "slice_4": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    self.weights_["16"],
                    self.weights_["17"],
                    self.weights_["18"],
                    self.weights_["19"],
                    self.weights_["20"],
                    self.weights_["21"],
                    self.weights_["22"],
                ]
            ),
            "slice_5": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    self.weights_["23"],
                    self.weights_["24"],
                    self.weights_["25"],
                    self.weights_["26"],
                    self.weights_["27"],
                    self.weights_["28"],
                    self.weights_["29"],
                ]
            ),
        }

    def call(self, input_: tf.Tensor) -> tf.Tensor:
        """Forward pass of the network.

        :param input_: the input image, expects shape [batch_size, sdim, sdim, 3]

        :returns: the output containing the intermediate activations.
        """
        h = self.network["slice_1"](input_)
        h_relu1_2 = h

        h = self.network["slice_2"](h)
        h_relu2_2 = h

        h = self.network["slice_3"](h)
        h_relu3_3 = h

        h = self.network["slice_4"](h)
        h_relu4_3 = h

        h = self.network["slice_5"](h)
        h_relu5_3 = h

        outputs = namedtuple(
            "Vgg16Outputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        return outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


class _SqueezeNet(tf.keras.Model):
    def __init__(self, parameter_path: str, weights_: dict = None) -> None:
        super(_SqueezeNet, self).__init__()

        self.parameter_path = os.path.join(parameter_path, "squeeze")

        if weights_ is None:
            self._load_architecture_from_file()
        else:
            self.weights_ = weights_
            self._initialise_architecture()

        self.n_slices = 7

    def _load_architecture_from_file(self):
        print("loading base parameters from file")
        self.network = {
            "slice_1": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_1_base.keras")
            ),
            "slice_2": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_2_base.keras")
            ),
            "slice_3": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_3_base.keras")
            ),
            "slice_4": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_4_base.keras")
            ),
            "slice_5": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_5_base.keras")
            ),
            "slice_6": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_6_base.keras")
            ),
            "slice_7": tf.keras.models.load_model(
                os.path.join(self.parameter_path, "slice_7_base.keras")
            ),
        }

    def _initialise_architecture(self):
        self.network = {
            "slice_1": tf.keras.Sequential(
                [
                    tf.keras.Input(
                        shape=(None, None, None)
                    ),  # something we have to do with tf...
                    self.weights_["0"],
                    self.weights_["1"],
                ]
            ),
            "slice_2": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    self.weights_["2"],
                    _Fire(
                        self.weights_["3"],
                        self.weights_["4"],
                        self.weights_["5"],
                        self.weights_["6"],
                        self.weights_["7"],
                        self.weights_["8"],
                    ),
                    _Fire(
                        self.weights_["9"],
                        self.weights_["10"],
                        self.weights_["11"],
                        self.weights_["12"],
                        self.weights_["13"],
                        self.weights_["14"],
                    ),
                ]
            ),
            "slice_3": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    self.weights_["15"],
                    _Fire(
                        self.weights_["16"],
                        self.weights_["17"],
                        self.weights_["18"],
                        self.weights_["19"],
                        self.weights_["20"],
                        self.weights_["21"],
                    ),
                    _Fire(
                        self.weights_["22"],
                        self.weights_["23"],
                        self.weights_["24"],
                        self.weights_["25"],
                        self.weights_["26"],
                        self.weights_["27"],
                    ),
                ]
            ),
            "slice_4": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    self.weights_["28"],
                    _Fire(
                        self.weights_["29"],
                        self.weights_["30"],
                        self.weights_["31"],
                        self.weights_["32"],
                        self.weights_["33"],
                        self.weights_["34"],
                    ),
                ]
            ),
            "slice_5": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    _Fire(
                        self.weights_["35"],
                        self.weights_["36"],
                        self.weights_["37"],
                        self.weights_["38"],
                        self.weights_["39"],
                        self.weights_["40"],
                    ),
                ]
            ),
            "slice_6": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    _Fire(
                        self.weights_["41"],
                        self.weights_["42"],
                        self.weights_["43"],
                        self.weights_["44"],
                        self.weights_["45"],
                        self.weights_["46"],
                    ),
                ]
            ),
            "slice_7": tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(None, None, None)),
                    _Fire(
                        self.weights_["47"],
                        self.weights_["48"],
                        self.weights_["49"],
                        self.weights_["50"],
                        self.weights_["51"],
                        self.weights_["52"],
                    ),
                ]
            ),
        }

    def call(self, input_: tf.Tensor) -> tf.Tensor:
        """Forward pass of the network.

        :param input_: the input image, expects shape [batch_size, sdim, sdim, 3]

        :returns: the output containing the intermediate activations.
        """
        h = self.network["slice_1"](input_)
        h_relu1 = h

        h = self.network["slice_2"](h)
        h_relu2 = h

        h = self.network["slice_3"](h)
        h_relu3 = h

        h = self.network["slice_4"](h)
        h_relu4 = h

        h = self.network["slice_5"](h)
        h_relu5 = h

        h = self.network["slice_6"](h)
        h_relu6 = h

        h = self.network["slice_7"](h)
        h_relu7 = h

        outputs = namedtuple(
            "SqueezeOutputs",
            ["relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7"],
        )
        return outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)


@tf.keras.utils.register_keras_serializable()  # makes importing more robust
class _Fire(tf.keras.Model):
    def __init__(
        self,
        squeeze: tf.keras.layers.Conv2D,
        squeeze_activation: tf.keras.layers.ReLU,
        expand1x1: tf.keras.layers.Conv2D,
        expand1x1_activation: tf.keras.layers.ReLU,
        expand3x3: tf.keras.layers.Conv2D,
        expand3x3_activation: tf.keras.layers.ReLU,
        **kwargs,
    ):
        super(_Fire, self).__init__(**kwargs)

        self.squeeze = squeeze
        self.squeeze_activation = squeeze_activation
        self.expand1x1 = expand1x1
        self.expand1x1_activation = expand1x1_activation
        self.expand3x3 = expand3x3
        self.expand3x3_activation = expand3x3_activation

    def get_config(self):
        base_config = super().get_config()
        config = {
            "squeeze": keras.saving.serialize_keras_object(self.squeeze),
            "squeeze_activation": keras.saving.serialize_keras_object(
                self.squeeze_activation
            ),
            "expand1x1": keras.saving.serialize_keras_object(self.expand1x1),
            "expand1x1_activation": keras.saving.serialize_keras_object(
                self.expand1x1_activation
            ),
            "expand3x3": keras.saving.serialize_keras_object(self.expand3x3),
            "expand3x3_activation": keras.saving.serialize_keras_object(
                self.expand3x3_activation
            ),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config["squeeze"] = keras.layers.deserialize(config["squeeze"])
        config["squeeze_activation"] = keras.layers.deserialize(
            config["squeeze_activation"]
        )
        config["expand1x1"] = keras.layers.deserialize(config["expand1x1"])
        config["expand1x1_activation"] = keras.layers.deserialize(
            config["expand1x1_activation"]
        )
        config["expand3x3"] = keras.layers.deserialize(config["expand3x3"])
        config["expand3x3_activation"] = keras.layers.deserialize(
            config["expand3x3_activation"]
        )
        return cls(**config)

    def call(self, input_: tf.Tensor) -> tf.Tensor:
        inter = self.squeeze_activation(self.squeeze(input_))
        return tf.concat(
            [
                self.expand1x1_activation(self.expand1x1(inter)),
                self.expand3x3_activation(self.expand3x3(inter)),
            ],
            -1,
        )


@tf.keras.utils.register_keras_serializable()  # makes importing more robust
class _NetLinLayer(tf.keras.Model):
    def __init__(
        self,
        conv_layer: tf.keras.layers.Conv2D,
        dropout_layer: tf.keras.layers.Dropout = None,
        **kwargs,
    ):
        super(_NetLinLayer, self).__init__(**kwargs)

        self.conv_layer = conv_layer
        self.dropout_layer = dropout_layer

    def get_config(self):
        base_config = super().get_config()
        config = {
            "conv_layer": keras.saving.serialize_keras_object(self.conv_layer),
            "dropout_layer": keras.saving.serialize_keras_object(self.dropout_layer),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        config["conv_layer"] = keras.layers.deserialize(config["conv_layer"])
        config["dropout_layer"] = keras.layers.deserialize(config["dropout_layer"])
        return cls(**config)

    def call(self, input_: tf.Tensor) -> tf.Tensor:
        """Forward pass of the network.

        :param input_: the input image, expects shape [batch_size, sdim, sdim, 3]

        :returns: the output tensor.
        """
        if self.dropout_layer is not None:
            return self.conv_layer(input_)
        else:
            return self.conv_layer(self.dropout_layer(input_))
