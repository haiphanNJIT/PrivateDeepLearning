# Private Deep Learning under Model Attacks

# AdLM, pCDBN, dp-Autoencoder, and StoBatch

These are the codes used in the papers titled: (1) Adaptive Laplace Mechanism: Differential Privacy Preservation in Deep Learning (https://arxiv.org/abs/1709.05750), (2) Preserving Differential Privacy in Convolutional Deep Belief Networks (https://arxiv.org/abs/1706.08839), and (3) Differential Privacy Preservation for Deep Auto-Encoders: an Application of Human Behavior Prediction (https://dl.acm.org/citation.cfm?id=3016005). 

# Requirement
The software is written in tensorflow. It requires the following packages:

python3

Tensorflow 1.1 or later

# How it works
Compute differentially private LRP for MNIST:

python dpLRP_MNIST.py 

Compute differentially private LRP for Cifar10:

python3 dpLRP_Cifar10.py 

Run AdLM on MNIST:

python3 AdLM.py

Run AdLM on Cifar10:

python3 AdLMCNN_CIFAR.py

Run evaluation of AdLM on Cifar10:

python3 CifarEval_AdLM2.py

Run pCDBN:

python3 pcdbn.py

Run dp-Autoencoder:

python3 dpautoencoder.py

Run dpSGD on MNIST:

python3 DPSGD_CNN.py

Run dpSGD on Cifar10:

python3 pSGDCNN_CIFAR.py

Run dpSGD evaluation on Cifar10:

python3 CifarEval.py

The default hyper-parameters can be edited in each main() file.

# StoBatch (Stochastic Batch Mechanism)

Code for the paper "Scalable Differential Privacy with Certified Robustness in Adversarial Learning" in ICML2020, https://128.84.21.199/pdf/1903.09822.pdf

# Requirements:
Python 3.x (tested with 3.5.2)

Tensorflow (1.x, tested on 1.15.0), numpy, scipy, imageio

An early (compatible) version of Cleverhans library is included

GPU with at least 11GB of memory, better to have at least 4 GPUs

Script should download the CIFAR10 automatically, if the dataset is not there

Raw TinyImageNet dataset https://tiny-imagenet.herokuapp.com/

Pretrained weight of resnet-18 ("resnet18_imagenet_1000_no_top.h5") from: https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5

# How it works:
For centralized MNIST and CIFAR-10: python3 StoBatch.py. The parameters can be finetuned in the main python file: StoBatch.py

For StoBatch with CIFAR10, run "StoBatch_cifar10.py" to train the model, run "StoBatch_cifar10_testing.py" to test.
Things to change before running: GPU settings, path to checkpoints, path to results.

For StoBatch with Tiny ImageNet:
Before running the training script, you can run the "tinyimagenet_read.py" to generate the pre-packed dataset (normalized and augmented). You need to change the path to data in that script. By default we use 30 out of 200 classes in the TinyImageNet.
    
For StoBatch with TinyImageNet, run "StoBatch_resnet_pretrain.py" to train the model, run "StoBatch_resnet_pretrain_testing.py" to test.

For SecureSGD (baseline method), run "SecureSGD_resnet_pretrain.py" to train the model, run "SecureSGD_resnet_pretrain_testing.py" to test.
    
Things you need to change before running: GPU settings (according to your own GPU setup), path to data, path to checkpoints

Things you can change (if you know it is necessary): Settings for training and testing, The weight mapping in: resnet18_weight_table_enc_layer.txt

About the "parameter_settings.txt": This described the core parameters we used in the code. How the parameters were defined in the code was kind of messy, so I made a summary there.

# Cites
If you use this code, please cite the following papers:

[1] Scalable Differential Privacy with Certified Robustness in Adversarial Learning. 
NhatHai Phan, My T. Thai, Han Hu, Ruoming Jin, Tong Sun, and Dejing Dou. The 37th International Conference on Machine Learning (ICML'20), July 12 - 18, 2020. https://128.84.21.199/pdf/1903.09822.pdf

[2] Heterogeneous Gaussian Mechanism: Preserving Differential Privacy in Deep Learning with Provable Robustness. 
NhatHai Phan, Minh Vu, Yang Liu, Ruoming Jin, Xintao Wu, Dejing Dou, and My T. Thai. The 28th International Joint Conference on Artificial Intelligence (IJCAI'19), August 10-16, 2019, Macao, China. https://arxiv.org/pdf/1906.01444.pdf

[3] Adaptive Laplace Mechanism: Differential Privacy Preservation in Deep Learning
NhatHai Phan, Xintao Wu, Han Hu, Dejing Dou. IEEE ICDM'17, New Orleans, USA 18-21 November 2017. https://arxiv.org/abs/1709.05750.

[4] Preserving Differential Privacy in Convolutional Deep Belief Networks
NhatHai Phan, Xintao Wu, Dejing Dou. Machine Learning 2017, ECML-PKDD Journal Track, Skopje, Macedonia 18-22 Sep 2017. https://arxiv.org/abs/1706.08839. 

[5] Differential Privacy Preservation for Deep Auto-Encoders: an Application of Human Behavior Prediction
NhatHai Phan, Yue Wang, Xintao Wu, and Dejing Dou. Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI-16). https://dl.acm.org/citation.cfm?id=3016005
