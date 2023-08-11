import pandas as pd
import numpy as np
import os
# from matplotlib import pyplot as plt


def loadParams():
    if os.path.exists("weightOne.npy" and "biasOne.npy"
                      and "weightTwo.npy" and "biasTwo.npy"):
        weightOne = np.load("weightOne.npy")
        biasOne = np.load("biasOne.npy")
        weightTwo = np.load("weightTwo.npy")
        biasTwo = np.load("biasTwo.npy")
    else:
        weightOne, biasOne, weightTwo, biasTwo = initParams()
        saveParams(weightOne, biasOne, weightTwo, biasTwo)
    return weightOne, biasOne, weightTwo, biasTwo


def saveParams(weightOne, biasOne, weightTwo, biasTwo):
    np.save("weightOne.npy", weightOne)
    np.save("biasOne.npy", biasOne)
    np.save("weightTwo.npy", weightTwo)
    np.save("biasTwo.npy", biasTwo)


def initParams():
    weightOne = np.random.rand(10, 784) - 0.5
    biasOne = np.random.rand(10, 1) - 0.5
    weightTwo = np.random.rand(10, 10) - 0.5
    biasTwo = np.random.rand(10, 1) - 0.5
    return weightOne, biasOne, weightTwo, biasTwo


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
    for i in Z:
        if Z[i] > 0:
            break
        else:
            Z[i] = 0
    return Z


def softMax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


def forwardProp(X, weightOne, biasOne, weightTwo, biasTwo):
    # first layer
    Z_1 = np.dot(weightOne, X) + biasOne  # lin regression
    A_1 = reLu(Z_1)  # activation function
    # second layer
    Z_2 = np.dot(weightTwo, A_1) + biasTwo  # lin regression
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
    dZ_2 = A_2 - oneHot(Y)
    dWeightTwo = 1/m * np.dot(dZ_2, A_1.T)
    dBiasTwo = 1/m * np.sum(dZ_2, 2)
    dZ_1 = np.dot(weightTwo.T, dZ_2) * derivativeReLu(Z_1)
    dWeightOne = 1/m * np.dot(dZ_1, X.T)
    dBiasOne = 1/m * np.sum(dZ_1, 2)
    return dWeightOne, dBiasOne, dWeightTwo, dBiasTwo


def updateParams(weightOne, biasOne, weightTwo, biasTwo, dWeightOne, dBiasOne,
                 dWeightTwo, dBiasTwo, learningRate):
    weightOne = weightOne - learningRate * dWeightOne
    biasOne = biasOne - learningRate * dBiasOne
    weightTwo = weightTwo - learningRate * dWeightTwo
    biasTwo = biasTwo - learningRate * dBiasTwo
    return weightOne, biasOne, weightTwo, biasTwo


def gradientDecent(weightOne, biasOne, weightTwo, biasTwo,
                   pixels, labels, iterations, learningRate):
    for i in range(iterations):
        Z_1, A_1, Z_2, A_2 = forwardProp(pixels, weightOne, biasOne,
                                         weightTwo, biasTwo)

        dWeightOne, dBiasOne, dWeightTwo, dBiasTwo = backwardProp(Z_1, A_1,
                                                                  Z_2, A_2,
                                                                  pixels,
                                                                  labels)

        weightOne, biasOne, weightTwo, biasTwo = updateParams(
                                                        weightOne,
                                                        biasOne,
                                                        weightTwo,
                                                        biasTwo,
                                                        dWeightOne,
                                                        dBiasOne,
                                                        dWeightTwo,
                                                        dBiasTwo,
                                                        0.01)
    return weightOne, biasOne, weightTwo, biasTwo


pixels, labels = prepareData()
weightOne, biasOne, weightTwo, biasTwo = loadParams()
weightOne, biasOne, weightTwo, biasTwo = gradientDecent(weightOne, biasOne,
                                                        weightTwo, biasTwo,
                                                        pixels, labels,
                                                        2000, 0.1)
saveParams(weightOne, biasOne, weightTwo, biasTwo)
