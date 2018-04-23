# Tensorflow_Deep_Taylor_LRP
Layerwise Relevance Propagation with Deep Taylor Series in TensorFlow.

You can use LRP to visualize the relative feature importances of the input to a neural network. 

## How to Use

### Step 1: Construct your tensorflow graph
### Step 2: Make sure your prediction layer (output layer) is named "absolute_output"
### Step 3: Make sure your input layer (shaped as [num_batches, height, width, num_channels]) is named "absolute_input"
### Step 4: `relevance_heatmap = lrp.lrp(prediction*label, lowest_value, highest_value)`
