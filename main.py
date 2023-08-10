import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

weightOne = np.random.rand(10, 784)
biasOne = np.random.rand(10, 1)
weightTwo = np.random.rand(10, 10)
biasTwo = np.random.rand(10, 1)


def loadParams():
    global weightOne, biasOne, weightTwo, biasTwo
    with open("params.txt") as params:
        lines = params.readlines()
        weightOne = float(lines[0])
        biasOne = float(lines[1])
        weightTwo = float(lines[2])
        biasTwo = float(lines[3])


def saveParams():
    with open("params.txt") as params:
        params[0] = weightOne
        params[1] = biasOne
        params[2] = weightTwo
        params[3] = biasTwo


def prepareData():
    trainingData = pd.read_csv('mnist_train.csv', header=None)
    trainingData = np.array(trainingData)
    np.random.shuffle(trainingData)  # randomize data to avoid overfit
    trainingData = trainingData.T  # transpose the matrix
    trainingData = trainingData[0:785, 0:2000]  # take only 2000 images
    rows, columns = trainingData.shape  # m = 785, n = 60000
    print(rows, columns)
    labels = trainingData[0]  # 1 x 60000 labels 
    pixels = trainingData[1:columns]  # pixels is a 784 x 60000 matrix, 784 pixels per image, 60000 images
    return labels, pixels


def reLu(Z):
    for i in Z:
        if Z[i] > 0:
            break
        else:
            Z[i] = 0
    return Z


def softMax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


def forwardProp(X):
    global weightOne, biasOne, weightTwo, biasTwo
    # first layer
    Z_1 = np.dot(weightOne, X) + biasOne
    A_1 = reLu(Z_1)
    # second layer
    Z_2 = np.dot(weightTwo, A_1) + biasTwo
    A_2 = softMax(Z_2)
    return Z_1, A_1, Z_2, A_2


def oneHot(Y):
    oneHot_Y = np.zeros((Y.size, Y.max() + 1))
    for i in Y:
        oneHot_Y[y] = 1


def backwardProp(Z_1, A_1, Z_2, A_2, Y):
    Y = oneHot(Y)
    dZ_2 = A_2 - Y


labels, pixels = prepareData()
Z_1, A_1, Z_2, A_2 = forwardProp(pixels)
