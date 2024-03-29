import numpy as np

from proj_helpers import *
from preprocess import *
from feature_engineer import *
from implementations import *

data_folder = 'data/'
train_data_path = data_folder + 'train.csv'
test_data_path = data_folder + 'test.csv'

print('Loading data\r')
y_data, tX_data, ids = load_csv_data(train_data_path)
_, tX_test_data, ids_test = load_csv_data(test_data_path)

models = ['gd',
          'sgd',
          'least squares',
          'ridge regression',
          'logistic regression',
          'regularized logistic regression']

#Chosing a model
selected_model = models[3]

#Hyper parameters
max_iters = 50
poly_degrees = [10, 5, 10, 6]
inv_log_degrees = [13, 19, 20, 15]
gammas = [0.001, 0.001, 0.001, 0.001]
lambdas = [1e-12, 1e-12, 1e-12, 1e-12]

# Masks creation
tX = tX_data
tX_test = tX_test_data
mask = get_categorical_masks(tX)
mask_test = get_categorical_masks(tX_test)
y_pred = np.zeros(len(tX_test))

print('Training with {}'.format(selected_model))
for idx in mask:
    print('Preprocessing mask {}'.format(idx))

    #training
    xtrain = tX[mask[idx]]
    ytrain = y_data[mask[idx]]

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
    xtrain_poly_exp = degree_expansion(xtrain_centered, poly_degrees[idx])
    xtrain_inv_log, val_col = invert_log(xtrain_no_constant, inv_log_degrees[idx])
    inv_log_stand, mean_log, std_log = standardize(xtrain_inv_log[:, 1:])
    #test
    xtest_poly_exp = degree_expansion(xtest_centered, poly_degrees[idx])
    xtest_inv_log = fit_invert_log(xtest_no_constant, val_col, inv_log_degrees[idx])
    inv_log_stand_test = fit_standardize(xtest_inv_log[:, 1:], mean_log, std_log)

    print('Fit model for masks {}'.format(idx))
    xtrain_process = np.column_stack((xtrain_poly_exp, xtrain_inv_log, inv_log_stand))

    if selected_model == 'gd':
        w_init = init_w(xtrain_process)
        w, loss = least_square_GD(ytrain, xtrain_process, initial_w=w_init, max_iters=max_iters, gamma=gammas[idx])
    elif selected_model == 'sgd':
        w_init = init_w(xtrain_process)
        w, loss = least_square_SGD(ytrain, xtrain_process, initial_w=w_init, max_iters=max_iters, gamma=gammas[idx])
    elif selected_model == 'least squares':
        w, loss = least_square(ytrain, xtrain_process)
    elif selected_model == 'ridge regression':
        w, loss = ridge_regression(ytrain, xtrain_process, lambdas[idx])
    elif selected_model == 'logistic regression':
        w_init = init_w(xtrain_process)
        ytrain = convert_y_to_log(ytrain)
        w, loss = logistic_regression(ytrain, xtrain_process, initial_w=w_init, max_iters=max_iters, gamma=gammas[idx])
        w = w.flatten()
    elif selected_model == 'regularized logistic regression':
        w_init = init_w(xtrain_process)
        ytrain = convert_y_to_log(ytrain)
        w, loss = reg_logistic_regression(ytrain, xtrain_process, lambda_ = lambdas[idx] , initial_w=w_init, max_iters=max_iters, gamma=gammas[idx])
        w = w.flatten()
    else:
        raise ValueError('Invalid model key')

    print('Minimal loss for masks {}: {}'.format(idx, loss))

    print('Accuracy of masks {}: {}%'.format(idx, np.mean(ytrain == predict_labels(w, xtrain_process))*100))

    xtest_process = np.column_stack((xtest_poly_exp, xtest_inv_log, inv_log_stand_test))
    y_pred_temp = predict_labels(w, xtest_process)

    #Add prediction of mask idx to all predictions
    y_pred[mask_test[idx]] = y_pred_temp

    print('------------------------------------------------------------------------------')

print('Training complete')

print('Writing submission\r')
OUTPUT_PATH = 'sample-submission.csv' # TODO: fill in desired name of output file for submission
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print('Done! Have a nice day :)')


