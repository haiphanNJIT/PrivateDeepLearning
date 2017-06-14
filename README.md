# Copyright 2017 Hai Phan, NJIT. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================================================

Differential Privacy Preservation in Deep Learning

1)	In the code the following parameters need to be defined:
D; #Data size#
numHidUnits; #Number of hidden units in one convolution layer#
numFeatures; #Number of features in one convolution layer#
Delta = 2*numHidUnits*numFeatures; #Function sensitivity#
epsilon; #Privacy budget epsilon#
loc, scale1 = 0., Delta*numHidUnits/(epsilon*D); #0-mean and variant of noise#
W_conv1Noise = np.random.laplace(loc, scale1, #number_of_parameters); #This is the latest version of W_conv1Noise#
W_conv1Noise = np.reshape(W_conv1Noise, shape); #This is the latest version of W_conv1Noise, where shape is the shape of inputs, i.e., images#

2)	The model is constructed similar with a regular model: 2 convolution layers, 1/2 fully connected layer, and a softmax layer. Run the CifarEval.py separately from the other code to get the prediction accuracy on testing data for the Cifar-10 dataset.
