from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from sklearn.preprocessing import StandardScaler
from NativeAnn import NeuralNet
import pickle
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#SETUP -- loading data, scaler and initializing classifier neural net
infile = open('mnist_weights_final','rb')
weights_batches_final = pickle.load(infile)
infile.close()

infile = open('mnist_biases_final','rb')
biases_final = pickle.load(infile)
infile.close()

infile = open('test_images','rb')
test_images = pickle.load(infile)
infile.close()

infile = open('test_labels','rb')
test_labels = pickle.load(infile)
infile.close()


infile = open('scaler','rb')
scaler = pickle.load(infile)
infile.close()

MNIST_classifier = NeuralNet('multi_logloss')
MNIST_classifier.addLayer(64, 'reLu')
MNIST_classifier.addLayer(64, 'reLu')
MNIST_classifier.addLayer(10, 'softmax')
MNIST_classifier.set_weights(weights_batches_final)
MNIST_classifier.set_biases(biases_final)

def getPred(pixel_matrix):
    pred = MNIST_classifier.predict(pixel_matrix)
    return np.argmax(pred, 0)

@app.route('/api/getAcc')
def get_test_accuracy():
    y_pred = MNIST_classifier.predict(test_images)
    test_accuracy = 100*np.sum(test_labels == np.argmax(y_pred, 0), axis=0) / test_images.shape[1]
    return str(test_accuracy)

@app.route('/')
def root():
    return "Hello World!"

@app.route('/api/classify', methods=['POST'])
@cross_origin()
def classify():
    pixel_matrix = np.array(request.json["pixelMatrix"]).reshape(1, 784)
    pixel_matrix = pixel_matrix/255
    outfile = open('random_img', 'wb')
    pickle.dump(pixel_matrix.reshape(784, 1), outfile)
    outfile.close()
    pred = getPred(pixel_matrix.reshape(784, 1)).tolist()
    return jsonify({'pred': pred})

if __name__ == '__main__':
    app.run()