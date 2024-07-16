## Setup
This repository was developed on Ubuntu 20.04 LTS using miniconda as the environment manager. 

Create the environment using the environment.yml file.
```sh
conda env create -f environment.yml
```

## Usage
### Generate test data (optional)
Navigate to the data/ directory and run the generate_test_data.py script.
```py
python generate_test_data.py -p
```
Note the -p switch will plot the images before saving them (disabled by default).

### Convert LPIPS from PyTorch to Tensorflow
From this directory, run the convert_lpips_torch2tf.py script.
```py
python convert_lpips_torch2tf.py -b 'all' -t -v
```
Note the base argument can either be 'alex', 'vgg16', 'squeeze', or 'all' (corresponding to running the conversion for all three available bases). The -t switch will run tests of the conversion (disabled by default). The -v switch will provide verbose output in testing (disabled by default).

The main output of this script are the trained parameters under parameters/ that can then be used in downstream tensorflow projects. Remember to re-run the generate_package.sh script in the parent directory if there's been changes to the parameters/loss function.

## Directory Structure
```
data/           the testing data for the conversion.
loss_fns/       the loss function definitions.
parameters/     the saved model parameters.
utils/          utility functions/classes.
```
