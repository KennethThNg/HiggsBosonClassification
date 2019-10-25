import numpy as np

from proj_helpers import *
from preprocess import *
from feature_engineer import *
from implementation import *

data_folder = '../data/'
train_data_path = data_folder + 'train.csv'
test_data_path = data_folder + 'test.csv'

print('load data:')
y, tX, ids = load_csv_data(train_data_path)
_, tX_test, ids_test = load_csv_data(test_data_path)

models = ['gd',
          'sgd',
          'least squares',
          'ridge regression']

selected_model = models[3]

#Hyper parameters
max_iters = 50
gamma = 0.1
poly_degree = 12
inv_log_degree = 6
gamma = 0.01
lambda_ = 0.001

# Masks creation
mask = get_categorical_masks(tX)
mask_test = get_categorical_masks(tX_test)
y_pred = np.zeros(len(tX_test))

print('Training with {}'.format(selected_model))
for idx in mask:
    print('Preprocessing mask {}'.format(idx))

    #training
    xtrain = tX[mask[idx]]
    ytrain = y[mask[idx]]

    xtrain_no_constant, col_to_remove = remove_constant_col(xtrain)
    xtrain_median = replaceNaNsWithMedian(xtrain_no_constant)
    xtrain_centered, mean, std = standardize(xtrain_median)

    #test
    xtest = tX_test[mask_test[idx]]
    xtest_no_constant = fit_remove_constant_col(xtest, col_to_remove)
    xtest_median = replaceNaNsWithMedian(xtest_no_constant)
    xtest_centered = fit_standardize(xtest_median, mean, std)

    print('Feature engineering for mask {}'.format(idx))
    #training
    xtrain_poly_exp = degree_expansion(xtrain_centered, poly_degree)
    xtrain_inv_log, val_col = invert_log(xtrain_no_constant, inv_log_degree)
    inv_log_stand, mean_log, std_log = standardize(xtrain_inv_log[:, 1:])
    #test
    xtest_poly_exp = degree_expansion(xtest_centered, poly_degree)
    xtest_inv_log = fit_invert_log(xtest_no_constant, val_col, inv_log_degree)
    inv_log_stand_test = fit_standardize(xtest_inv_log[:, 1:], mean_log, std_log)

    print('Fit model for masks {}'.format(idx))
    xtrain_process = np.column_stack((xtrain_poly_exp, xtrain_inv_log, inv_log_stand))

    if selected_model == 'gd':
        w_init = init_w(xtrain_process)
        w, loss = least_square_GD(ytrain, xtrain_process, initial_w=w_init, max_iters=max_iters, gamma=gamma)
    elif selected_model == 'sgd':
        w_init = init_w(xtrain_process)
        w, loss = least_square_SGD(ytrain, xtrain_process, initial_w=w_init, max_iters=max_iters, gamma=gamma)
    elif selected_model == 'least squares':
        w, loss = least_square(ytrain, xtrain_process)
    elif selected_model == 'ridge regression':
        w, loss = ridge_regression(ytrain, xtrain_process, lambda_)
    else:
        raise ValueError('Invalid model key')

    print('Minimal loss for masks {}: '.format(idx, loss))

    print('Accuracy of masks {}: {}%'.format(idx, np.mean(ytrain == predict_labels(w, xtrain_process))*100))

    xtest_process = np.column_stack((xtest_poly_exp, xtest_inv_log, inv_log_stand_test))
    y_pred_temp = predict_labels(w, xtest_process)

    y_pred[mask_test[idx]] = y_pred_temp

    print('------------------------------------------------------------------------------')



print('...Training complete')


