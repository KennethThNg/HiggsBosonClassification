import numpy as np
from helpers import*

def least_square_GD(y_o, tx, initial_w, max_iters, gamma):
    y = y_o.reshape((len(y_o), 1))
    w = initial_w
    for i in range(max_iters):
        loss = compute_loss(y, tx, w)
        grad = compute_gradient(y, tx, w)
        w = w - gamma*grad
    return w, loss

def least_square_SGD(y_o, tx, initial_w, max_iters, gamma, batch_size=1):
    y = y_o.reshape((len(y_o), 1))
    w = initial_w
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            loss = compute_loss(y_batch, tx_batch, w)
            grad = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma*grad
    return w, loss

def least_square(y, tx):
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    mu = lambda_/2*len(y)
    A = tx.T.dot(tx) + mu*np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(A,b)
    loss = compute_loss(y, tx, w)
    return w, loss

def logistic_regression(y_o, tx, initial_w, max_iters, gamma, batch_size=1):
    y = y_o.reshape((len(y_o),1))
    w = initial_w
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            loss = compute_logistic_loss(y_batch, tx_batch, w)
            grad = compute_gradient_logistic(y_batch, tx_batch, w)
            w = w - gamma*grad
    return w, loss

def reg_logistic_regression(y_o, tx, lambda_, initial_w, max_iters, gamma, batch_size=1):
        y = y_o.reshape((len(y_o),1))
    w = initial_w
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            loss = compute_logistic_loss(y_batch, tx_batch, w, lambda_)
            grad = compute_gradient_logistic(y_batch, tx_batch, w)
            w = w - gamma*grad
    return w, loss
