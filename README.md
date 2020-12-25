An app allowing users to draw a number and an artificial neural network trained on the MNIST data set will attempt to predict the number.

### Neural Network Framework Used
A geenral neural network framework developed in https://github.com/MustafaAli789/native-ann is employed in this flask/react project (checkout the General-Neural-Network.ipynb to see the different tests performed on the network). A two hidden layer, fully connected neural network is at play here. Both hidden layers use the 'reLu' activation function and have 64 neurons each. The output layer contains 10 neurons and utilzies the 'softmax' activation function to provide probabilities amongst 10 options. 

The input for predictions to the network is a numpy 2d matrix with 784 columns and 1 row (i.e 28x28 pixel black and white images reshaped into a column vector). 

### Live Website
The application is live at https://mnist-native-classifier.herokuapp.com/. 
You can check out https://mnist-native-classifier.herokuapp.com/api/accuracy to see the accuracy of the network against some test data (10k images).
There is also a POST endpoint at https://mnist-native-classifier.herokuapp.com/api/classify (used interally as well) that can be used to predict a handwriten digit in the form of a 2d array (784x1, does not have to be a numpy array). The response from the server will contain 'pred' for the final prediction and an array 'preds' containig individual probabilities.

### Some notes
The network has the hardest time descerning the following numbers in order from most confused to least:
9 --> 7 --> 1
The rest of the numbers can be quite easily predicted but these 3 (especially the first 2) can be quite tricky at times. A probable hypothesis behind this could be the fact that the network was trained on a dataset composed of handwritten digits, that is, written by hand whereas in this case, the numbers are digitally drawn by mouse/trackpad. This could result in less 'natural' looking drawn digits compared to the more 'flowy' digits in the dataset; however, this is just a theory and not verified. 

Regardless of these faults, the network works decently at descerning between the rest of the digits.
