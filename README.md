# Private Deep Learning

# AdLM, pCDBN, and dp-Autoencoder

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

# Cites
If you use this code, please cite the following papers:

Adaptive Laplace Mechanism: Differential Privacy Preservation in Deep Learning
NhatHai Phan, Xintao Wu, Han Hu, Dejing Dou. https://arxiv.org/abs/1709.05750.

Preserving Differential Privacy in Convolutional Deep Belief Networks
NhatHai Phan, Xintao Wu, Dejing Dou. https://arxiv.org/abs/1706.08839. 

Differential Privacy Preservation for Deep Auto-Encoders: an Application of Human Behavior Prediction
NhatHai Phan, Yue Wang, Xintao Wu, and Dejing Dou. Proceedings of the 30th AAAI Conference on Artificial Intelligence (AAAI-16).
