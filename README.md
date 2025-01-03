# URV MESIIA 2024-2025 AVPR Assignment 2

Universitat Rovira i Virgilii, Tarragona.

Masters in Computer Security and Artificial Intelligence.

Artificial Vision and Pattern Recognition course.

January 2025

Mark Safrónov

This repository contains the implementation for the Assignment 2 of the Artificial Vision and Pattern Recognition course.

It is not self-sufficient, you must obtain the [Stanford dataset](http://vision.stanford.edu/Datasets/40actions.html) first to run the code here.

All the paths expected by the code are in the [my_paths.py](./my_paths.py) module.
You must obtain the JPEGImages, XMLAnnotations and ImageSplits parts of the dataset.

The task is described in the [Task2.pdf](./Task2.pdf).

The work protocol is described in the Jupyter notebooks.
Their names are prefixed with numbers to sort naturally in the order of running.

1. [Create the train:test splits](./00-creating-train-test-splits.ipynb)
2. [Analyse the data (just some statistics for reference)](./00-data-analysis.ipynb)
3. [Run the Custom CNN validation array](./01-raw-images-custom.ipynb)
4. [Run the ResNet validation array](./01-raw-images-resnet.ipynb)
5. [Run the ResNet with 4-channel images with inbuilt bounding boxes](./02-bbox-resnet-augmented.ipynb)

All the Python modules in this folder which has `my_` prefixes are the core Python code used in the Jupyter notebooks.

The most interesting are:

1. [my_models](./my_models.py) - definitions for the models used
2. [my_transforms](./my_transforms.py) - the transforms used for the dataset
3. [my_train_model](./my_train_model.py) - training protocol
4. [my_test_model](./my_test_model.py) - testing protocol

In addition to that, a custom dataset implementation was made to generate the 4-channel images with inbuilt bounding boxes on-the-fly: [ImageFolderWithBBox](./ImageFolderWithBBox.py).
The code inside it was tested in an auxiliary notebook [test-bitmask-transform](./test-bitmask-transform.ipynb).

The final deliverable is [2024-12-31 Mark Safronov - AVPR 2.pdf](./2024-12-31%20Mark%20Safronov%20-%20AVPR%202.pdf)