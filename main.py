import pandas as pd
import numpy as np
import os
# from matplotlib import pyplot as plt


def loadParams():
    if os.path.exists("W_1.npy" and "b_1.npy"
                      and "W_2.npy" and "b_2.npy"):
        W_1 = np.load("W_1.npy")
        b_1 = np.load("b_1.npy")
        W_2 = np.load("W_2.npy")
        b_2 = np.load("b_2.npy")
    else:
        W_1, b_1, W_2, b_2 = initParams()
        saveParams(W_1, b_1, W_2, b_2)
    return W_1, b_1, W_2, b_2


def saveParams(W_1, b_1, W_2, b_2):
    np.save("W_1.npy", W_1)
    np.save("b_1.npy", b_1)
    np.save("W_2.npy", W_2)
    np.save("b_2.npy", b_2)


def initParams():
    W_1 = np.random.rand(10, 784) - 0.5
    b_1 = np.random.rand(10, 1) - 0.5
    W_2 = np.random.rand(10, 10) - 0.5
    b_2 = np.random.rand(10, 1) - 0.5
    return W_1, b_1, W_2, b_2


def prepareData():
    trainingData = pd.read_csv('mnist_train.csv', header=None)
    trainingData = np.array(trainingData)
    np.random.shuffle(trainingData)  # randomize data to avoid overfit
    trainingData = trainingData.T  # transpose the matrix
    trainingData = trainingData[0:785, 0:2000]  # take only 2000 images
    rows, columns = trainingData.shape  # m = 785, n = 60000
    print(rows, columns)
    labels = trainingData[0]  # 1 x 60000 labels
    pixels = trainingData[1:columns]  # pixels is a 784 x 60000 matrix, 784
    # pixels per image, 60000 images
    return labels, pixels


def reLu(Z):
    return np.maximum(Z, 0)


def softMax(Z):
    exp = np.exp(Z - np.max(Z)) 
    return exp / exp.sum(axis=0)


def forwardProp(X, W_1, b_1, W_2, b_2):
    # first layer
    Z_1 = np.dot(W_1, X) + b_1  # lin regression
    A_1 = reLu(Z_1)  # activation function
    # second layer
    Z_2 = np.dot(W_2, A_1) + b_2  # lin regression
    A_2 = softMax(Z_2)  # propapility normalization
    return Z_1, A_1, Z_2, A_2


def oneHot(Y):
    oneHot_Y = np.zeros((Y.size, Y.max() + 1))
    oneHot_Y[np.arange(Y.size), Y] = 1
    oneHot_Y = oneHot_Y.T
    return oneHot_Y


# derivative of reLu function is 1, if Z > 0, else 0
def derivativeReLu(Z):
    return Z > 0


def backwardProp(Z_1, A_1, Z_2, A_2, X, Y):
    m = Y.size
    oneHot_Y = oneHot(Y)
    dZ_2 = 2 * (A_2 - oneHot_Y)
    dW_2 = 1/m * np.dot(dZ_2, A_1.T)
    db_2 = 1/m * np.sum(dZ_2, 1)
    dZ_1 = np.dot(W_2.T, dZ_2) * derivativeReLu(Z_1)
    dW_1 = 1/m * np.dot(dZ_1, X.T)
    db_1 = 1/m * np.sum(dZ_1, 1)
    return dW_1, db_1, dW_2, db_2


def updateParams(W_1, b_1, W_2, b_2, dW_1, db_1,
                 dW_2, db_2, learningRate):
    W_1 = W_1 - learningRate * dW_1
    b_1 = b_1 - learningRate * np.reshape(db_1, (10, 1))
    W_2 = W_2 - learningRate * dW_2
    b_2 = b_2 - learningRate * np.reshape(db_2, (10, 1))
    return W_1, b_1, W_2, b_2


def getPrediction(A_2):
    return np.argmax(A_2, 0)


def getAccuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradientDecent(W_1, b_1, W_2, b_2,
                   pixels, labels, iterations, learningRate):
    for i in range(iterations):
        Z_1, A_1, Z_2, A_2 = forwardProp(pixels, W_1, b_1,
                                         W_2, b_2)

        dW_1, db_1, dW_2, db_2 = backwardProp(Z_1, A_1, Z_2, A_2,
                                              pixels, labels)
        if (i % 10 == 0):
            print("iteration: ", i) 
            predictions = getPrediction(A_2)
            print(getAccuracy(predictions, labels))

        W_1, b_1, W_2, b_2 = updateParams(W_1, b_1, W_2, b_2,
                                          dW_1, db_1, dW_2, db_2,
                                          learningRate)
    return W_1, b_1, W_2, b_2


labels, pixels = prepareData()
W_1, b_1, W_2, b_2 = loadParams()
W_1, b_1, W_2, b_2 = gradientDecent(W_1, b_1, W_2, b_2,
                                    pixels, labels,
                                    100, 0.1)
saveParams(W_1, b_1, W_2, b_2)
