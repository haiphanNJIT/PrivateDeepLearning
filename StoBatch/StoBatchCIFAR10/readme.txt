Code for paper "Scalable Differential Privacy with Certified Robustness in Adversarial Learning" in ICML2020
Author: NhatHai Phan, Han Hu
For CIFAR10 with conv net

Requirements:
    Python 3.x (tested with 3.5.2)
    Tensorflow (1.x, tested on 1.15.0), numpy, scipy, imageio
    An early (compatible) version of Cleverhans library is included
    GPU with at least 11GB of memory, better to have at least 4
    Script should download the CIFAR10 automatically, if the dataset is not there

Instructions:
    For StoBatch with CIFAR10, run "StoBatch_cifar10.py" to train the model, run "StoBatch_cifar10_testing.py" to test.
    Things to change before running: 
        GPU settings
        path to checkpoints
        path to results