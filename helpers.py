import numpy as np
from proj_helpers import predict_labels
import matplotlib.pyplot as plt

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return e.T.dot(e)/(2*len(e))

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
    """apply sigmoid function on t."""
    s = 1 / (1 + np.exp(-t))
    s[s < 1e-8] = 1e-8
    s[s > 0.99999999] = 0.99999999
    return s

def compute_gradient_logistic(y, tx, w, lambda_=0):
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) + lambda_ * w
    return grad

def compute_logistic_loss(y, tx, w, lambda_=0):
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) - (1-y).T.dot(np.log(1 - pred)) + lambda_ * (w.T @ w) / 2
    return loss

# Find best model parameter
def find_best_parameters(grid_results, min_ = False, item = 'acc_mean'):
    max = -10000
    min = 10000
    index = -1
    for i, res in enumerate(grid_results):
        feature = res[item]
        if min_ and feature < min:
            min = feature
            index = i
        elif not min_ and feature > max:
            max = feature
            index = i
    if min_:
        return min, index
    else:
        return max, index
    
# visualized dependence of accuracy on lambda
def accuracy_visualization_1(lambds, acc, param1='param1', title='Accuracy Plot'):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure()
    plt.semilogx(lambds, acc, marker=".", color='b', label='train error')
    plt.xlabel(str(param1))
    plt.ylabel("acc")
    plt.title(str(title))
    plt.grid(True)
    plt.savefig('plots/' + str(title) + '.png')
    plt.show()
    
# visualized the matrice 'accs' on a grid made from vectors 'param1' and 'param2'
def accuracy_visualization_2(param_x, param_y, accs, name_x='param1', name_y='param2', title='Accuracy Visualization'):
    X, Y = np.meshgrid(param_x, param_y, sparse=True)
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    c = ax.pcolor(X, Y, accs, cmap='RdBu', vmin=np.min(accs), vmax=np.max(accs))
    ax.set_title(str(title))
    ax.set_xlabel(str(name_x))
    ax.set_ylabel(str(name_y))
    fig.colorbar(c, ax=ax)
    fig.savefig('plots/' + str(title) + '.png')
    plt.show()
