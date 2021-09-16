import numpy as np


class NeuralNework():

    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.alpha = alpha
        self.W = []

        # 2 - 2 - 1 with bias
        # x00(bias)       x10(bias)
        # x01             x11              y
        # x02             x12
        for i in range(0, len(layers) - 2):
            # number of weights in layer would be number of hidden unit in current layer * number of hidden unit in
            # next layer (add bias if we need trainable bias)
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # for last two layers

        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "Neural Network : {}  ".format("_".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=100, displayUpdate=100):
        # add bias in last column
        print(X.shape)
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in range(epochs):
            for (x, target) in zip(X, y):
                self.partialFit(x, target)

            if epoch == 0 or epoch % displayUpdate == 0:
                loss = self.calculateLoss(X, y)
                print("[INFO] epoch : {} loss {:.7f} ".format(epoch + 1, loss))

    def partialFit(self, X, y):

        # FORWARD PASS
        # initialize as input ans pass to the network
        activationsList = [np.atleast_2d(X)]
        for layer in range(0, len(self.W)):
            net = activationsList[layer].dot(self.W[layer])
            activationFunc = self.sigmoid(net)
            activationsList.append(activationFunc)

        # Now A is updated with forward pass data.

        # BACKWARD PASS
        # get the error
        # error is last activations minus the ground truth label
        error = activationsList[-1] - y

        # build delta (gradient delta list)
        D = [error * self.sigmoid_derivative(activationsList[-1])]

        # iterate reverse

        for layer in range(len(activationsList) - 2, 0, -1):
            grad_delta = D[-1].dot(self.W[layer].T)
            grad_delta = grad_delta * self.sigmoid_derivative(activationsList[layer])
            D.append(grad_delta)

        D = D[::-1]

        # UPDATE PASS
        for layer in range(len(self.W)):
            self.W[layer] += -self.alpha * activationsList[layer].T.dot(D[layer])

    def calculateLoss(self, X, y):
        #  could have been any loss let's say we are taking squared loss here
        y = np.atleast_2d(y)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - y) ** 2)
        return loss

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        # FORWARD PASS
        for layer in range(len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        return p
