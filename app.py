from flask import Flask, jsonify, request
from sklearn.preprocessing import StandardScaler
from NativeAnn import NeuralNet
import pickle
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True

#SETUP -- loading data, scaler and initializing classifier neural net
infile = open('mnist_weights','rb')
weights_batches = pickle.load(infile)
infile.close()

infile = open('scaler','rb')
scaler = pickle.load(infile)
infile.close()

MNIST_classifier = NeuralNet('multi_logloss')
MNIST_classifier.addLayer(64, 'reLu')
MNIST_classifier.addLayer(10, 'softmax')
MNIST_classifier.set_weights(weights_batches)

def getPred(pixel_matrix):
    pred = MNIST_classifier.predict(pixel_matrix.reshape(784, 1))
    return np.argmax(pred, 0)

@app.route('/')
def root():
    return "Hello World!"

@app.route('/api/classify', methods=['POST'])
def classify():
    pixel_matrix = np.array(request.json["pixelMatrix"])
    #num_array = scaler.transform(num_array)
    pred = getPred(pixel_matrix).tolist()
    return jsonify({'pred': pred})

if __name__ == '__main__':
    app.run()