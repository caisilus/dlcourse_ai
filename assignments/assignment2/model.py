import numpy as np
from numpy.core.fromnumeric import argmax

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        #Create necessary layers
        self.fully_conected_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.W1 = self.fully_conected_1.params()['W']
        self.B1 = self.fully_conected_1.params()['B']
        self.fully_conected_2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.W2 = self.fully_conected_2.params()['W']
        self.B2 = self.fully_conected_2.params()['B']
        self.relu = ReLULayer()


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        
        # Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for param_key in self.params():
            param = self.params()[param_key]
            param.grad = np.zeros_like(param.value)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        layer1_res = self.fully_conected_1.forward(X)
        relu_res = self.relu.forward(layer1_res)
        layer2_res = self.fully_conected_2.forward(relu_res)
        loss, d_out = softmax_with_cross_entropy(layer2_res, y)
        d_out /= X.shape[0]
        loss /= X.shape[0]

        d_layer2 = self.fully_conected_2.backward(d_out)
        d_relu = self.relu.backward(d_layer2)
        d_layer1 = self.fully_conected_1.backward(d_relu)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        loss_reg_W1, grad_reg_W1 = l2_regularization(self.W1.value, self.reg)
        loss_reg_B1, grad_reg_B1 = l2_regularization(self.B1.value, self.reg)
        loss_reg_W2, grad_reg_W2 = l2_regularization(self.W2.value, self.reg)
        loss_reg_B2, grad_reg_B2 = l2_regularization(self.B2.value, self.reg)
        
        loss_reg = loss_reg_W1 + loss_reg_B1 + loss_reg_W2 + loss_reg_B2
        loss += loss_reg

        self.W1.grad += grad_reg_W1
        self.W2.grad += grad_reg_W2
        self.B1.grad += grad_reg_B1
        self.B2.grad += grad_reg_B2

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        layer1_res = self.fully_conected_1.forward(X)
        relu_res = self.relu.forward(layer1_res)
        layer2_res = self.fully_conected_2.forward(relu_res)
        softmax_res = softmax(layer2_res)

        pred = argmax(softmax_res, axis=1)
        return pred

    def params(self):
        #result = {}

        # Implement aggregating all of the params

        result = {'W1': self.W1, 'B1': self.B1, 'W2': self.W2, 'B2': self.B2}

        return result
