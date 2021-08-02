import numpy as np

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # implement softmax
    # Your final implementation shouldn't have any loops
    pred = predictions.copy().T
    pred -= np.max(pred, axis=0)
    
    exp_pred = (np.exp(pred))
    sum_exp = np.sum(exp_pred, axis=0)
    probs = (exp_pred / sum_exp).T

    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # implement cross-entropy
    # Your final implementation shouldn't have any loops
    grid = np.indices(probs.shape)
    mask = (grid[-1].T == target_index.flatten()).T
    loss_vec = probs[mask] #x where p(x) = 1
    loss = -np.sum(np.log(loss_vec))
    return loss
  

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    
    dprediction = np.zeros_like(predictions)

    #target_index = target_index.flatten()
    #print(target_index)

    grid = np.indices(predictions.shape)
    mask = (grid[-1].T == target_index.flatten()).T
    #print(mask)
    
    pred = predictions.copy().T
    pred -= np.max(pred, axis=0)

    exp_pred = (np.exp(pred))
    sum_exp = np.sum(exp_pred, axis=0)

    pred = pred.T

    pgt = pred[mask]
    #print(pgt)
    e_pgt = np.exp(pgt)

    dprediction = (exp_pred / sum_exp).T
    c = sum_exp - e_pgt
    dprediction[mask] = -c / sum_exp

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X
        relu_fun = np.vectorize((lambda x: x if x > 0 else 0))
        return relu_fun(X)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # Implement backward pass
        # Your final implementation shouldn't have any loops
        mask_fun = np.vectorize((lambda x: 1 if x > 0 else 0))
        mask = mask_fun(self.X)

        d_result = mask * d_out
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        res = np.dot(X, self.W.value) + self.B.value
        return res

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        ones_flat = np.ones(d_out.shape[0])
        ones = ones_flat[::,None]
        X = np.concatenate((self.X, ones), axis=1)
        dWB = np.dot(X.T, d_out)
        dW = dWB[:-1,:]
        self.W.grad = dW
        dB = (dWB[-1,:])[None,::]
        self.B.grad = dB

        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
