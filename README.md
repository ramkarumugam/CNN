# CNN
convolutional neural network, simplest implementation for understanding. 

This repository is a simple and quick implementation of the CNN using one convolution and on hidden layer network. 

The code includes bias, momentum learning and learning rate decay. Beginners or students can use their dataset and check how this works. 

The dataset used can be found in "http://cogcomp.org/Data/Car/". All copyrights related to the dataset must be adhered.
This is for learning purpose only

The file 'Read.m' reads all images and stores in to an image array. 
Then run 'Start.m', which is the executable file, which calls other programs and complete convolution and training. 
'Imageprocess.m' which will do convolution and pooling. The matrix obj.hor can be replaced with your desired convolution kernel. 
'trainer.m' - training algorithm. 
'NeuralNetwork.m' - A three layer, neural network with bias and momentum learning. 
