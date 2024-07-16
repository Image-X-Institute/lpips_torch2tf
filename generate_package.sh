#!/bin/bash

# Help #
Help()
{
    echo "Generate a package that can be copied into Tensorflow projects"
    echo
    echo "generate_package [-h]"
    echo "options:"
    echo "h     Print this help"
}

while getopts ":h" option; do
    case $option in 
        h) # Display help
            Help
            exit;;
        ?) # Invalid option
            echo "ERROR: Invalid option"
            exit;;
    esac
done

mkdir -p ./package/parameters
cp -r ./dev_src/parameters/lpips_tf ./package/parameters/lpips_tf

mkdir -p ./package/loss_fns
cp ./dev_src/loss_fns/lpips_base_tf.py ./package/loss_fns

