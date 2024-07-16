# Lpips Torch to Tensorflow

**Authors** James Grover and David E.J. Waddington

[Learned Perceptual Image Patch Similarity (LPIPS)](https://arxiv.org/abs/1801.03924) is a popular loss function that can be used to train neural networks. It utilises a clever use of pre-trained vision neural networks to extract deep feature differences rather than 'shallow' features differences extracted by MSE or SSIM. The authors of this loss function maintain a [PyTorch implementation](https://github.com/richzhang/PerceptualSimilarity). We provide a Tensorflow (2.x) port.

## Basic Usage / Install
**Requires Python 3.10 or above.**

**The only dependency for using the loss function is Tensorflow 2.x.**

### Linux / MacOS
Run the generate_package.sh script to generate a copyable package to place in your Tensorflow project.
```sh
sh generate_package.sh
```

### Windows
Run the generate_package.bat script to generate a copyable package to place in your Tensorflow project.
```sh
generate_package.bat
```
Follow the prompts (note the destination is a directory).

### Linux / MacOS / Windows
Copy the contents of package/ into your Tensorflow project (don't copy the package itself, just what's inside of it).
Your directory structure should look something like this:
```
\a_tf_training_project
      \parameters
          alex\
              ...
          vgg16\
              ...
          squeeze\
              ...
      \loss_fns
          lpips_base_tf.py
      a_training_script_you_have_developed.py
      other_stuff_you_need...          
```

Import the loss function inside your Tensorflow script.
```py
from loss_fns import lpips_base_tf
```

Instantiate the loss function inside your Tensorflow script/notebook etc.
```py
loss_fn_lpips = lpips_base_tf.LPIPS(base='alex') # or 'vgg16' or 'squeeze'
```

**If your data is not normalised between [-1,1] you can instantiate LPIPS with pre_norm = True, this will normalise the data as required.**
```py
loss_fn_lpips = lpips_base_tf.LPIPS(base='alex', pre_norm=True)
```

Use the loss function!
```py
loss = loss_fn_lpips(some_RGB_image_in_NHWC_format, some_other_RGB_image_in_NHWC_format)
```

Examples are available in basic_demo.ipynb and end2end_demo.ipynb, including how to blend this loss function with other (pixelwise) loss functions. There are two dummy images located in dev_src/data/.

## Dev Usage

Follow the README located in dev_src/.

## IT DOESN'T WORK ON MY MACHINE!
We have tested our implementation on Ubuntu 20.04 and Windows 11. If you find this repository does not work on your particular build, please raise an issue with as many details as practical.
