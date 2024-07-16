import torch


def test_gpu() -> None:
    """Tests to see if a CUDA GPU can be retrieved by PyTorch."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda:0":
        print(
            f"GPU (cuda:0) found! \n Details: {torch.cuda.get_device_properties(device)}"
        )
    else:
        print(f"GPU (cuda:0) not found, check PyTorch installation.")


def test_lpips() -> None:
    """Tests to see if lpips can be imported and tests their loss fns"""
    try:
        import lpips
    except ImportError:
        print("lpips repository not found!")
    else:
        loss_fn_alex = lpips.LPIPS(net="alex")
        loss_fn_vgg = lpips.LPIPS(net="vgg")
        img0 = torch.rand(1, 3, 64, 64)
        img1 = torch.rand(1, 3, 64, 64)
        print(f"loss alex: {loss_fn_alex(img0, img1)}")
        print(f"loss vgg: {loss_fn_vgg(img0, img1)}")


def main() -> None:
    test_gpu()
    test_lpips()


if __name__ == "__main__":
    main()
