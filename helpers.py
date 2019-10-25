import numpy as np
from proj_helpers import predict_labels

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return e.dot(e)/(2*len(e))

def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e)/len(e)
    return grad

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def init_w(tx, seed=1):
    """Initializes w with random values in [0,1) based on shape of tx."""
    np.random.seed(seed)
    return np.random.rand(tx.shape[1])[:,np.newaxis]

def accuracy(y, x, w):
    """Computes the accuracy of the predictions."""
    return np.mean(y == predict_labels(w, x))

def sigmoid(t):
    return 1/(1+np.exp(-t))

def compute_gradient_logistic(y, tx, w, lambda_=0):
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) + 2*lambda_ * w
    return grad

def compute_logistic_loss(y, tx, w, lambda_=0):
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) - (1-y).T.dot(np.log(1 - pred)) + lambda_+np.squeeze(w.T.dot(w))
    return loss

