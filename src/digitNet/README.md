# digitNet 
This directory contains a lightweight network I trained to detect digits in a 32x32 window. 
The network consist of three convolutional layer with max pooling, and a fully conected output-
layer from which the second FC layer is the softmax classification result. 
The network is trained over the whole mnist dataset (including test and validation set) and
all non-italic digits from chars78k dataset. All images is of digits between [1-9]. I also
created some test samples myself from the git branch "data_gathererererer". The complete 
trainingset consisted of just under 70.000 images. 
The performance of running the network for 1000 epochs with adadelta optimizer and dataset
augmentation for regularization was actually surprisingly good considering how cheap the 
network is compared to some early iterations.

Additionally I've created some python scripts for getting the datasets, and "freezing" the 
network for use in OpenCV.
