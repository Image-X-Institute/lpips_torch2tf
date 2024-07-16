import torch
import torchvision

from collections import OrderedDict, namedtuple
import os


class LPIPS(torch.nn.Module):
    """Recreating the LPIPS class from the PerceptualSimilarity repository/PyPi package in base PyTorch.
    A few custom modifications are present.
    Credit: https://github.com/richzhang/PerceptualSimilarity
    """

    def __init__(
        self,
        base: str = "alex",
        base_weights: str = "DEFAULT",
        base_require_grad: bool = False,
        add_lpips_dropout: bool = True,
        lpips_pretrained: bool = True,
        eval_mode: bool = True,
        spatial: bool = True,
    ) -> None:
        """The LPIPS loss function.

        :param base: the base model, options are 'alex', 'vgg16', 'squeeze'.
        :param base_weights: the base model weights.
        :param base_require_grad: will the base be optimized?
        :param add_lpips_dropout: add lpips dropout?
        :param lpips_pretrained: load the pre-trained lpips parameters?
        :param eval_mode: turn the model to eval?
        :param spatial: add spatial averaging?
        """
        super(LPIPS, self).__init__()

        self.base = base
        self.base_weights = base_weights
        self.base_require_grad = base_require_grad

        self.add_lpips_dropout = add_lpips_dropout
        self.lpips_pretrained = lpips_pretrained

        self.eval_mode = eval_mode
        self.spatial = spatial

        self.scaling_layer = _ScalingLayer()

        self.parameter_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "parameters", "lpips_torch"
        )

        self._parse_base()
        self._initialise_lpips()

    def _parse_base(self):
        print(f"fetching base: {self.base}")
        match self.base.lower():
            case "alex":
                self.base_net = _AlexNet(self.base_weights, self.base_require_grad)
                self.chs = [64, 192, 384, 256, 256]
                self.model_path_name = "alex"
            case "vgg16":
                self.base_net = _VGG16Net(self.base_weights, self.base_require_grad)
                self.chs = [64, 128, 256, 512, 512]
                self.model_path_name = "vgg"
            case "squeeze":
                self.base_net = _SqueezeNet(self.base_weights, self.base_require_grad)
                self.chs = [64, 128, 256, 384, 384, 512, 512]
                self.model_path_name = "squeeze"
            case _:
                raise ValueError(
                    "Base net not supported, options are ['alex', 'vgg16', 'squeeze']"
                )

        self.L = len(self.chs)

    def _initialise_lpips(self):
        match self.base.lower():
            case "alex" | "vgg16":
                self.lin0 = _NetLinLayer(
                    self.chs[0], add_dropout=self.add_lpips_dropout
                )
                self.lin1 = _NetLinLayer(
                    self.chs[1], add_dropout=self.add_lpips_dropout
                )
                self.lin2 = _NetLinLayer(
                    self.chs[2], add_dropout=self.add_lpips_dropout
                )
                self.lin3 = _NetLinLayer(
                    self.chs[3], add_dropout=self.add_lpips_dropout
                )
                self.lin4 = _NetLinLayer(
                    self.chs[4], add_dropout=self.add_lpips_dropout
                )

                self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

                self.lins = torch.nn.ModuleList(self.lins)

            case "squeeze":
                self.lin0 = _NetLinLayer(
                    self.chs[0], add_dropout=self.add_lpips_dropout
                )
                self.lin1 = _NetLinLayer(
                    self.chs[1], add_dropout=self.add_lpips_dropout
                )
                self.lin2 = _NetLinLayer(
                    self.chs[2], add_dropout=self.add_lpips_dropout
                )
                self.lin3 = _NetLinLayer(
                    self.chs[3], add_dropout=self.add_lpips_dropout
                )
                self.lin4 = _NetLinLayer(
                    self.chs[4], add_dropout=self.add_lpips_dropout
                )
                self.lin5 = _NetLinLayer(
                    self.chs[5], add_dropout=self.add_lpips_dropout
                )
                self.lin6 = _NetLinLayer(
                    self.chs[6], add_dropout=self.add_lpips_dropout
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

                self.lins = torch.nn.ModuleList(self.lins)

            case (
                _
            ):  # Should already be caught by _parse_base but here for completeness.
                raise ValueError(
                    "Base net not supported, options are ['alex', 'vgg16', 'squeeze']"
                )

        if self.lpips_pretrained:
            print("loading pre-trained lpips parameters")
            model_path = os.path.join(
                self.parameter_path, self.model_path_name + ".pth"
            )
            _model = torch.load(model_path, map_location="cpu")
            tmp = (
                {}
            )  # A very hacky way to fix the 'network' 'model' naming discrepency.
            for key in _model.keys():
                tmp[f'{key.split('.')[0]}.network.{key.split('.')[2]}.{key.split('.')[3]}'] = _model[key]
            self.load_state_dict(tmp, strict=False)

        if self.eval_mode:
            self.eval()

    def _normalise(self, input_: torch.Tensor, type: str = "pre") -> torch.Tensor:
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
                    torch.sqrt(torch.sum(input_**2, dim=1, keepdim=True)) + 1e-10
                )

    def _upsample(
        self, input_: torch.Tensor, out_HW: tuple = (64, 64), mode: str = "bilinear"
    ) -> torch.Tensor:
        """Resizes the input tensor (image).

        :param input_: the input image to be resized.
        :param out_HW: the desired height/width of the returned image.
        :param mode: the interpolation mode.

        :returns: the resized image.
        """

        return torch.nn.Upsample(size=out_HW, mode=mode, align_corners=False)(input_)

    def _spatial_average(
        self, input_: torch.Tensor, keepdim: bool = True
    ) -> torch.Tensor:
        """Performs a spatial average over the last two dims.

        :param input_: the input image to be averaged.
        :param keepdim: whether or not to compress the out tensor.

        :returns: the spatially averaged image.
        """
        return input_.mean([2, 3], keepdim=keepdim)

    def _generate_features_and_differences(
        self, output_0: torch.Tensor, output_1: torch.Tensor
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

    def forward(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        retPerLayer: bool = False,
        normalise: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward function of the loss calculation.

        :param image_0: the reference image.
        :param image_1: the comparator image.
        :param retPerLayer: whether to return the residual per layer with the output.
        :param normalise: whether to normalise the data [-1, 1].
        """
        if normalise:
            image_0, image_1 = self._normalise(image_0), self._normalise(image_1)

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


class _ScalingLayer(torch.nn.Module):
    def __init__(self) -> None:
        super(_ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Implements a scaling for the input.

        :param input_: the input tensor to be scaled.

        :returns: the scaled tensor.
        """
        return (input_ - self.shift) / self.scale


class _AlexNet(torch.nn.Module):
    def __init__(self, weights: str = "DEFAULT", requires_grad: bool = False) -> None:
        super(_AlexNet, self).__init__()
        self.weights = weights
        self.requires_grad = requires_grad

        self.n_slices = 5

        self._fetch_parameters()
        self._initialise_architecture()

    def _fetch_parameters(self):
        self.pretrained_parameters = torchvision.models.alexnet(
            weights=self.weights
        ).features

    def _initialise_architecture(self):
        self.network = torch.nn.ModuleDict(
            {
                "slice_1": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("0", self.pretrained_parameters[0]),
                            ("1", self.pretrained_parameters[1]),
                        ]
                    )
                ),
                "slice_2": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("2", self.pretrained_parameters[2]),
                            ("3", self.pretrained_parameters[3]),
                            ("4", self.pretrained_parameters[4]),
                        ]
                    )
                ),
                "slice_3": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("5", self.pretrained_parameters[5]),
                            ("6", self.pretrained_parameters[6]),
                            ("7", self.pretrained_parameters[7]),
                        ]
                    )
                ),
                "slice_4": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("8", self.pretrained_parameters[8]),
                            ("9", self.pretrained_parameters[9]),
                        ]
                    )
                ),
                "slice_5": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("10", self.pretrained_parameters[10]),
                            ("11", self.pretrained_parameters[11]),
                        ]
                    )
                ),
            }
        )
        if not self.requires_grad:
            for parameter in self.network.parameters():
                parameter.requires_grad = False

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        :param input_: the input image, expects shape [batch_size, 3, sdim, sdim]

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


class _VGG16Net(torch.nn.Module):
    def __init__(self, weights: str = "DEFAULT", requires_grad: bool = False) -> None:
        super(_VGG16Net, self).__init__()
        self.weights = weights
        self.requires_grad = requires_grad

        self.n_slices = 5

        self._fetch_parameters()
        self._initialise_architecture()

    def _fetch_parameters(self):
        self.pretrained_parameters = torchvision.models.vgg16(
            weights=self.weights
        ).features

    def _initialise_architecture(self):
        self.network = torch.nn.ModuleDict(
            {
                "slice_1": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("0", self.pretrained_parameters[0]),
                            ("1", self.pretrained_parameters[1]),
                            ("2", self.pretrained_parameters[2]),
                            ("3", self.pretrained_parameters[3]),
                        ]
                    )
                ),
                "slice_2": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("4", self.pretrained_parameters[4]),
                            ("5", self.pretrained_parameters[5]),
                            ("6", self.pretrained_parameters[6]),
                            ("7", self.pretrained_parameters[7]),
                            ("8", self.pretrained_parameters[8]),
                        ]
                    )
                ),
                "slice_3": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("9", self.pretrained_parameters[9]),
                            ("10", self.pretrained_parameters[10]),
                            ("11", self.pretrained_parameters[11]),
                            ("12", self.pretrained_parameters[12]),
                            ("13", self.pretrained_parameters[13]),
                            ("14", self.pretrained_parameters[14]),
                            ("15", self.pretrained_parameters[15]),
                        ]
                    )
                ),
                "slice_4": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("16", self.pretrained_parameters[16]),
                            ("17", self.pretrained_parameters[17]),
                            ("18", self.pretrained_parameters[18]),
                            ("19", self.pretrained_parameters[19]),
                            ("20", self.pretrained_parameters[20]),
                            ("21", self.pretrained_parameters[21]),
                            ("22", self.pretrained_parameters[22]),
                        ]
                    )
                ),
                "slice_5": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("23", self.pretrained_parameters[23]),
                            ("24", self.pretrained_parameters[24]),
                            ("25", self.pretrained_parameters[25]),
                            ("26", self.pretrained_parameters[26]),
                            ("27", self.pretrained_parameters[27]),
                            ("28", self.pretrained_parameters[28]),
                            ("29", self.pretrained_parameters[29]),
                        ]
                    )
                ),
            }
        )
        if not self.requires_grad:
            for parameter in self.network.parameters():
                parameter.requires_grad = False

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        :param input_: the input image, expects shape [batch_size, 3, sdim, sdim]

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


class _SqueezeNet(torch.nn.Module):
    def __init__(self, weights: str = "DEFAULT", requires_grad: bool = False) -> None:
        super(_SqueezeNet, self).__init__()
        self.weights = weights
        self.requires_grad = requires_grad

        self.n_slices = 7

        self._fetch_parameters()
        self._initialise_architecture()

    def _fetch_parameters(self):
        self.pretrained_parameters = torchvision.models.squeezenet1_1(
            weights=self.weights
        ).features

    def _initialise_architecture(self):
        self.network = torch.nn.ModuleDict(
            {
                "slice_1": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("0", self.pretrained_parameters[0]),
                            ("1", self.pretrained_parameters[1]),
                        ]
                    )
                ),
                "slice_2": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("2", self.pretrained_parameters[2]),
                            ("3", self.pretrained_parameters[3]),
                            ("4", self.pretrained_parameters[4]),
                        ]
                    )
                ),
                "slice_3": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("5", self.pretrained_parameters[5]),
                            ("6", self.pretrained_parameters[6]),
                            ("7", self.pretrained_parameters[7]),
                        ]
                    )
                ),
                "slice_4": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("8", self.pretrained_parameters[8]),
                            ("9", self.pretrained_parameters[9]),
                        ]
                    )
                ),
                "slice_5": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("10", self.pretrained_parameters[10]),
                        ]
                    )
                ),
                "slice_6": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("11", self.pretrained_parameters[11]),
                        ]
                    )
                ),
                "slice_7": torch.nn.Sequential(
                    OrderedDict(
                        [
                            ("12", self.pretrained_parameters[12]),
                        ]
                    )
                ),
            }
        )
        if not self.requires_grad:
            for parameter in self.network.parameters():
                parameter.requires_grad = False

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        :param input_: the input image, expects shape [batch_size, 3, sdim, sdim]

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


class _NetLinLayer(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int = 1, add_dropout: bool = False
    ) -> None:
        super(_NetLinLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_dropout = add_dropout

        self._initialise_architecture()

    def _initialise_architecture(self):
        _conv_layer = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        if self.add_dropout:
            self.network = torch.nn.Sequential(torch.nn.Dropout(), _conv_layer)
        else:
            self.network = torch.nn.Sequential(_conv_layer)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        :param input_: the input image, expects shape [batch_size, 3, sdim, sdim]

        :returns: the output tensor.
        """
        return self.network(input_)
