# -*- coding: utf-8 -*-
'''
An implementation of Differentially Private Layer-wise Relevance Propagation (dpLRP) for MNIST dataset.
Author: Hai Phan, CCS, NJIT.
'''

import matplotlib.pyplot as plt
import numpy as np ; na = np.newaxis
import pickle;
import model_io
import data_io
import os
#import render

#load a neural network, as well as the MNIST test data and some labels
nn = model_io.read(os.getcwd() + '/models/MNIST/LeNet-5.txt') # 99.23% prediction accuracy
X = data_io.read(os.getcwd() + '/data/MNIST/train_images.npy')
Y = data_io.read(os.getcwd() + '/data/MNIST/train_labels.npy')

#print(Y);
# transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
X =  X / 127.5 - 1.

#reshape the vector representations in X to match the requirements of the CNN input
X = np.reshape(X,[X.shape[0],28,28,1])
X = np.pad(X,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))

# transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
I = Y[:,0].astype(int)
Y = np.zeros([X.shape[0],np.unique(Y).size])
Y[np.arange(Y.shape[0]),I] = 1
print(Y);

#permute data order for demonstration. or not. your choice.
I = np.arange(X.shape[0])
I = np.random.permutation(I)

index = 0;
R = np.random.laplace(0.0, 0.1, 32 * 32);
R = np.reshape(R, [1, 32, 32, 1]);
#predict and perform LRP for the 2000 first samples
Final_AvgR = [];
for i in I[:2000]:
    x = X[i:i+1,...]

    #forward pass and prediction
    ypred = nn.forward(x)

    #compute first layer relevance according to prediction
    #R = nn.lrp(ypred)                   #as Eq(56) from DOI: 10.1371/journal.pone.0130140
    if index == 0:
        R = nn.lrp(ypred,'epsilon',1.)    #as Eq(58) from DOI: 10.1371/journal.pone.0130140
    else:
        R += nn.lrp(ypred,'epsilon',1.);
    index += 1;

#############################
##Hyper-parameter Setting####
#############################
image_size = 28;
d = image_size**2; #The number of coefficients/features in each image
epsilon = 0.075; #epsilon for dpLRP
D = 60000; #size of the dataset
#############################

########################################
##Compute Differentially Private LRP####
########################################
#Average LRP#
Final_AvgR = R/(index + 1);

#Compute min and max in LRP#
min_R = 100000;
max_R = -100000;
for k in range(0,image_size+4): #padding size is 4#
   for j in range(0, image_size+4): #padding size is 4#
       if Final_AvgR[0,k,j] > max_R:
           max_R = Final_AvgR[0,k,j];
       if Final_AvgR[0,k,j] < min_R:
           min_R = Final_AvgR[0,k,j];
###

#Normalize and Perturb LRP to enforce Differential Privacy#
for k in range(0,image_size+4): #padding size is 4#
    for j in range(0, image_size+4): #padding size is 4#
        #(1) normalized LRP with inflation rate is 2, and (2) add Laplace noise (2*d/epsilon*D)
        Final_AvgR[0,k,j] = (Final_AvgR[0,k,j] - min_R)**4/(max_R - min_R)**4 + np.random.laplace(0.0, 2*d/(epsilon*D));
    #print(Final_AvgR[0,k,14]);
########################################

#Export dpLRP#
with open(os.getcwd() + '/Relevance_R_0_075.txt', "w") as text_file:
    for k in range(0, image_size+4): #padding size is 4#
        for j in range(0, image_size+4): #padding size is 4#
            text_file.write(str(Final_AvgR[0][k][j][0]) + "\t");
        text_file.write("\n");
    text_file.close();
##############
