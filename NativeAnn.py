import numpy as np
from scipy.special import expit


class Layer:

    # Activation Functions
    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        return 1 - np.square(np.tanh(x))

    def sigmoid(self, x):
        return expit(x)

    def d_sigmoid(self, x):
        return (1 - self.sigmoid(x)) * self.sigmoid(x)

    def ReLu(self, z):
        return np.maximum(0, z)

    def d_ReLu(self, Z):
        return Z > 0

    # For output layer, useful for multiclass classification
    def softmax(self, Z):
        return np.exp(Z) / sum(np.exp(Z))

    def d_softmax(self, Z):
        pass

    activationFunctions = {
        'tanh': (tanh, d_tanh),
        'sigmoid': (sigmoid, d_sigmoid),
        'reLu': (ReLu, d_ReLu),
        'softmax': (softmax, d_softmax)
    }

    # Input -> num of neurons in prev layer, Neurons --> num neurons in cur layer, Activation -> activation fxn to use
    def __init__(self, inputs, neurons, activation):
        self.neurons = neurons
        self.W = np.random.rand(neurons, inputs) - 0.5
        self.b = np.random.rand(neurons, 1) - 0.5
        self.Z = None
        self.A_prev = None
        self.act, self.d_act = self.activationFunctions.get(activation)

    def initializeWeights(self, inputs, neurons):
        self.W = np.random.rand(neurons, inputs) - 0.5

    def getNeuronCount(self):
        return self.neurons

    def getWeights(self):
        return self.W

    def getBias(self):
        return self.b

    def setWeight(self, weight):
        self.W = weight

    def setBias(self, bias):
        self.b = bias

    def feedForward(self, A_prev):
        # ipdb.set_trace()
        self.A_prev = A_prev
        self.Z = self.W.dot(self.A_prev) + self.b
        self.A = self.act(self, self.Z)
        return self.A

    # All derivatives are wrt to cost
    # Expects dA of cur layer
    # Special case where doing multi class classification with mutli class logloss, you can get the dZ wrt dC directly without having to first get dA
    def backprop(self, dA, learning_rate, dZ_Special):
        # ipdb.set_trace()

        # elementt by element matrix multip, not a normal dot prod since both matrices have same shape (essentialyl scalar)
        dZ = np.multiply(self.d_act(self, self.Z), dA) if dZ_Special.any() == None else dZ_Special

        # need to normalize weights and divide by number of samples
        # because it is actually a sum of weights
        dW = 1 / dZ.shape[1] * np.dot(dZ, self.A_prev.T)

        # this is to match shape since biases is supposed to be a col vector with 1 col but dZ has m cols
        # w/ m being num of samples, we want to take avg of all samples in dZ (i.e on a row by row basis, sum of cols
        # and divide by total num of smamples)
        db = 1 / dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)

        dA_prev = np.dot(self.W.T, dZ)

        self.W = self.W - learning_rate * dW
        self.b = self.b - learning_rate * db
        return dA_prev


class NeuralNet:

    # Loss Functions, mse for regression, logloss for classification
    def mse(self, a, target):
        return np.square(a - target)

    def d_mse(self, a, target):
        return 2 * (a - target)

    def binary_logloss(self, a, target):
        return -(target * np.log(a) + (1 - target) * np.log(1 - a))

    def d_binary_logloss(self, a, target):
        return (a - target) / (a * (1 - a))

    # Source - https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/discussion/2644
    def multi_logloss(self, a, target, eps=1e-15):
        predictions = np.clip(a, eps, 1 - eps)

        # normalize row sums to 1
        predictions /= predictions.sum(axis=1)[:, np.newaxis]

        return -np.sum(target * np.log(predictions)) / predictions.shape[0]

    def d_multi_logloss(self, a, target):
        return np.zeros(a.shape)  # kinda just a placeholder

    lossFunctions = {
        'mse': (mse, d_mse),
        'binary_logloss': (binary_logloss, d_binary_logloss),
        'multi_logloss': (multi_logloss, d_multi_logloss)
    }

    # LossFunction is either mse of logloss
    def __init__(self, lossFunction):
        self.layers = []
        self.learning_rate = 0.1
        self.epochs = 100
        self.batch_size = 10
        self.classification = False if lossFunction == 'mse' else True
        self.lossFunction = lossFunction
        self.loss, self.d_loss = self.lossFunctions.get(lossFunction)

    # Units is 1-n and activationFunction is 'ReLu', 'sigmoid', 'tanh', or 'softmax'
    def addLayer(self, units, activationFunction):
        prevLayerNeuronCount = self.layers[-1].getNeuronCount() if len(self.layers) > 0 else 0
        self.layers.append(Layer(prevLayerNeuronCount, units, activationFunction))

    def getNumBatches(self, num_samples, batch_size):
        if (num_samples == batch_size):
            return 1
        elif (num_samples > batch_size):
            if (num_samples % batch_size == 0):
                return num_samples // batch_size
            else:
                return (num_samples // batch_size) + 1
        else:
            return 1

    def oneHot(self, x):
        one_hot_X = np.zeros((x.max() + 1, x.size))  # making a matrix of 10 x m
        one_hot_X[x, np.arange(
            x.size)] = 1  # going through all cols and setting the row w/ index corresponding to the y to 1, its very easy to iterate over numpy arays like this apparently
        return one_hot_X

    # Convert one hot encoded 2d array to original array of 1d
    def rev_one_hot(self, target):
        rev_one_hot = np.argmax(target, 0)
        return rev_one_hot

    # Compare two 1d arrays
    def get_accuracy(self, target, Y, accuracy_buffer):
        return np.sum(abs(target - Y) < accuracy_buffer) / Y.size

    def get_layer_biases(self):
        biases = []
        for layer in self.layers:
            biases.append(layer.getBias())
        return biases

    def get_layer_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.getWeights())
        return weights

    # Pass weights array, num of elemtns in weights array has to match up w/ layers and num of neurons in layers
    def set_weights(self, weights):
        if (len(weights) != len(self.layers)):
            raise ValueError("Num of layers and num of weihts must match")
        for count, weight in enumerate(weights):
            if (weight.shape[0] != self.layers[count].getNeuronCount()):
                raise ValueError(
                    "Num of rows in weights at index " + count + " does not match num of neurons in layer " + count)
        for count, weight in enumerate(weights):
            self.layers[count].setWeight(weight)

    def set_biases(self, biases):
        if (len(biases) != len(self.layers)):
            raise ValueError("Num of layers and num of biases must match")
        for count, bias in enumerate(biases):
            if (bias.shape[0] != self.layers[count].getNeuronCount()):
                raise ValueError(
                    "Num of rows in biases at index " + count + " does not match num of neurons in layer " + count)
        for count, bias in enumerate(biases):
            self.layers[count].setBias(bias)

    def fit(self, X, y, epochs=None, batch_size=None, learning_rate=None, accuracy_buffer=0.1):
        self.learning_rate = learning_rate if learning_rate != None else self.learning_rate
        self.epochs = epochs if epochs != None else self.epochs
        self.batch_size = batch_size if batch_size != None else self.batch_size

        # Need at min one layer
        if (len(self.layers) == 0):
            raise ValueError('No layers have been added. Need at least one layer. Please add a layer')

            # multi class classificaiton problem need y to be one hot encoded and must use multi log loss
        multiClassProblem = self.classification and (y.max() - y.min() > 1)
        if (multiClassProblem):
            y = self.oneHot(y)
            if (self.lossFunction != 'multi_logloss'):
                raise ValueError('Loss Function Must be multi_logloss for multi class classification')

        epoch_costs = []
        batches_cost_sum = 0
        num_batches = self.getNumBatches(X.shape[1], self.batch_size)

        # Initializing weights of the first layer
        # Need to do it right now because shape of input isnt known until now
        self.layers[0].initializeWeights(X.shape[0], self.layers[0].getNeuronCount())

        ###-----Epoch iterations, training occurs here-----###
        for epoch in range(self.epochs):
            batches_cost_sum = 0
            for batch in range(num_batches):

                ###-----Obtaining appropriate batch data-----###
                A = X[:, batch * self.batch_size:(batch + 1) * self.batch_size]

                if (multiClassProblem):
                    y_curBatch = y[:, batch * self.batch_size:(batch + 1) * self.batch_size]
                else:
                    y_curBatch = y[batch * self.batch_size:(batch + 1) * self.batch_size]

                ###-----Performing forward prop and backprop-----###
                # ipdb.set_trace()
                for layer in self.layers:
                    A = layer.feedForward(A)
                batches_cost_sum += 1 / self.batch_size * np.sum(self.loss(self, A, y_curBatch))

                # For multi class classiifcaiton problems (class > 2) and using softmax, deriv of softmax w.r.t to Zfinal is just actual - pred
                dZ_Special = A - y_curBatch if multiClassProblem else np.array([None])

                # After the final output layer dA is found like this since A is just the output
                dA = self.d_loss(self, A, y_curBatch)

                # Only final layer does the special dZ matter and only if multi class
                for layer in reversed(self.layers):
                    if (layer == self.layers[-1]):
                        dA = layer.backprop(dA, self.learning_rate, dZ_Special)
                    else:
                        dA = layer.backprop(dA, self.learning_rate, np.array([None]))

                ###-----Logging Metrics-----###
                if (epoch % 10 == 0 and batch == num_batches - 1):
                    print("-----Epoch: ", epoch, "-----")
                    if (multiClassProblem):
                        A = self.rev_one_hot(A)
                        y_curBatch = self.rev_one_hot(y_curBatch)
                    print("Accuracy:", self.get_accuracy(A, y_curBatch, accuracy_buffer))
                    print("Cost:", batches_cost_sum)
            epoch_costs.append(batches_cost_sum)
        return epoch_costs, self.get_layer_weights(), self.get_layer_biases()

    def predict(self, X):
        A = X
        for layer in self.layers:
            A = layer.feedForward(A)
        return A
