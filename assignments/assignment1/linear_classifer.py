import numpy as np
import math

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


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
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


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W) # num_batch x classes

    # implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    loss, grad = softmax_with_cross_entropy(predictions, target_index)

    dW = np.dot(X.T, grad)
    
    loss /= X.shape[0]
    dW /= X.shape[0]

    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            
            #print(batches_indices)
            # Don't forget to add both cross-entropy loss
            # and regularization!
            # implement generating batches from indices
            for batch_indices in batches_indices:
                batch_X = X[batch_indices,:]
                batch_y = y[batch_indices]
                # Compute loss and gradients
                loss_classifier, dW_classifier = linear_softmax(batch_X, self.W, batch_y) 
                loss_reg, dW_reg = l2_regularization(self.W, reg)
                loss = loss_classifier + loss_reg
                loss_history.append(loss)
                dW = dW_classifier + dW_reg
                # Apply gradient to weights using learning rate
                self.W = self.W - learning_rate * dW
            
            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        predictions = np.dot(X, self.W)
        probs = softmax(predictions)
        y_pred = np.argmax(probs, axis=1)

        # Implement class prediction
        # Your final implementation shouldn't have any loops

        return y_pred



                
                                                          

            

                
