import torch
from utils.dummys import torch_model, torch_dataset


def test_with_generated_data(
    loss_fn_pypi: torch.nn.Module,
    loss_fn_base: torch.nn.Module,
    image_0: torch.Tensor,
    image_1: torch.Tensor,
) -> None:
    """Tests whether the loss function gives the correct output based on pre-generated data.

    :param loss_fn_pypi: the loss function from the lpips PyPi package.
    :param loss_fn_base: the loss function that was re-created in this repository.
    :param image_0: an image to test (expects it to be grayscale [1, sdim, sdim]).
    :param image_1: an image to test (expects it to be grayscale [1, sdim, sdim]).
    """
    print("running tests on generated data...")
    print(f'PyPi loss output: {loss_fn_pypi(
        image_0.repeat(3,1,1).unsqueeze(0),
        image_1.repeat(3,1,1).unsqueeze(0)
        )
    }')
    print(f'PyTorch base loss output: {loss_fn_base(
        image_0.repeat(3,1,1).unsqueeze(0),
        image_1.repeat(3,1,1).unsqueeze(0)
        )
    }')
    print(f'Tensors equal?: {torch.equal(
            loss_fn_pypi(
                image_0.repeat(3,1,1).unsqueeze(0),
                image_1.repeat(3,1,1).unsqueeze(0)
            ),
            loss_fn_base(
                image_0.repeat(3,1,1).unsqueeze(0),
                image_1.repeat(3,1,1).unsqueeze(0)
            )
        )
    }')
    print(88 * "-")


def test_with_random_data(
    loss_fn_pypi: torch.nn.Module, loss_fn_base: torch.nn.Module, num_test: int = 100
) -> None:
    """Tests whether the loss function gives the correct output based on randomly generated data.

    :param loss_fn_pypi: the loss function from the lpips PyPi package.
    :param loss_fn_base: the loss function that was re-created in this repository.
    :param num_test: The number of random tests to run.
    """
    print("running tests with random data")
    res = 0
    for _ in range(num_test):
        in_1 = torch.randn(4, 1, 48, 64)
        in_2 = torch.randn(4, 1, 48, 64)
        res += torch.equal(loss_fn_pypi(in_1, in_2), loss_fn_base(in_1, in_2))

    print(f"{res} out of {num_test} tests were equal")
    print(88 * "-")


def integration_test(
    loss_fn_base: torch.nn.Module, image_0: torch.Tensor, image_1: torch.Tensor
) -> None:
    """Integration test showing the loss can work in a training loop.

    :param loss_fn_base: the loss function that was re-created in this repository.
    :param image_0: a label image to test (expects it to be grayscale [1, sdim, sdim] - it will be converted to RGB).
    :param image_1: an input image to test (expects it to be grayscale [1, sdim, sdim] - it will be converted to RGB).
    """
    print("commencing integration test")
    architecture = torch_model.DummyModel()
    optimizer = torch.optim.Adam(architecture.parameters())
    dataset = torch_dataset.DummyDataset(image_1, image_0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        prediction = architecture(inputs)
        try:
            loss = torch.mean(loss_fn_base(prediction, labels))  # need to take mean.
            print(f"loss computed without error, loss value: {loss}")
        except Exception as e:
            print("something went wrong with loss.backward()")
            print(e)
        try:
            loss.backward()
            print("loss.backward() computed without error")
        except Exception as e:
            print("something went wrong with loss.backward()")
            print(e)
        optimizer.step()
    print("integration test complete")
    print(88 * "-")
