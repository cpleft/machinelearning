"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  
"""
import numpy as np
import random

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 随机生成高斯分布
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """ "a" is an input to the network, return output """
        for i in range(len(self.sizes)-2):
            a = sigmoid(np.dot(self.weights[i], a)+self.biases[i])
        return softmax(np.dot(self.weights[-1], a)+self.biases[-1])

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                softmax_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def SGD(self, training_data, epochs, eta, test_data=None):
        """Train the neural network using stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            stochastic = training_data[0];
            # update_stochastic(eta)
            x = np.hsplit(stochastic, [self.sizes[0]])[0].reshape(self.sizes[0], 1)
            y = np.hsplit(stochastic, [self.sizes[0]])[1].reshape(len(stochastic) \
                    - self.sizes[0], 1)
            nabla_b, nabla_w = self.backprop(x, y)
            self.weights = [w - eta*nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - eta*nb
                           for b, nb in zip(self.biases, nabla_b)]


        if test_data:
            for i in range(n_test):
                x = np.array(test_data[i]).reshape(len(test_data[i]), 1)
                y = self.feedforward(x)
                s = 0
                for j in range(4):
                    if y[s] < y[j+1]:
                        s = j+1
                s += 1
                print(s)

        '''
        print(self.weights)
        print(self.biases)
        '''


def sigmoid(z):
    return 1.0/(1.0+np.exp(z))
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
def softmax(z):
    ez = np.exp(z)
    denominator = 0.0
    for i in ez:
        denominator += i
    return ez/denominator
def softmax_prime(z):
    return softmax(z)



if __name__ == "__main__":
# Identify number "1", "2", "3", "4", "5" by BP ANN
    # training_data
    X = [[0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0],
         [1,1,1,1,0, 0,0,0,0,1, 0,1,1,1,0, 1,0,0,0,0, 1,1,1,1,1],
         [1,1,1,1,0, 0,0,0,0,1, 0,1,1,1,0, 0,0,0,0,1, 1,1,1,1,0],
         [0,0,0,1,0, 0,0,1,1,0, 0,1,0,1,0, 1,1,1,1,1, 0,0,0,1,0],
         [1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,0, 0,0,0,0,1, 1,1,1,1,0]]
    Y = [[1,0,0,0,0],
         [0,1,0,0,0],
         [0,0,1,0,0],
         [0,0,0,1,0],
         [0,0,0,0,1]]

    training_data = np.hstack((X,Y))
    # test_data
    X_test = [[0,0,1,1,0, 0,0,1,1,0, 0,1,0,1,0, 0,0,0,1,0, 0,1,1,1,0],
              [1,1,1,1,0, 0,0,0,0,1, 0,1,1,1,0, 1,0,0,0,1, 1,1,1,1,1],
              [1,1,1,1,0, 0,0,0,0,1, 0,1,1,1,0, 1,0,0,0,1, 1,1,1,1,0],
              [0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0],
              [0,1,1,1,1, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 1,1,1,1,0]]

    net = Network([25, 50, 5])
    net.SGD(training_data, 1000, 0.01, X)

