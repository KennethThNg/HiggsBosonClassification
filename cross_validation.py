from preprocess import *
from feature_engineer import *
from implementation import *

def build_k_indices(y, k_fold, seed=1):
    """Builds k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]

    return np.array(k_indices)


def cross_validation(ytrain, xtrain, ytest, xtest, initial_w, max_iter, gamma, lambda_, model='ridge_regression'):
    if model == 'gd':
        w, loss = least_square_GD(ytrain, xtrain, initial_w, max_iter, gamma)
    elif model == 'sgd':
        w, loss = least_square_SGD(ytrain, xtrain, initial_w, max_iter, gamma)
    elif model == 'least squares':
        w, loss = least_square(ytrain, xtrain)
    elif model == 'ridge regression':
        w, loss = ridge_regression(ytrain, xtrain, lambda_)
    elif model == 'logistic regression':
        w, loss = logistic_regression(ytrain, xtrain, initial_w, max_iter, gamma)
    elif model == 'regularized logistic regression':
        w, loss = reg_logistic_regression(ytrain, xtrain, lambda_, initial_w, max_iter, gamma)
    else:
        raise ValueError('Invalid model')

    ypred = predict_labels(w, xtest)

    acc = (ypred == ytest).mean()

    return w, acc

def cross_validation_demo(y, tx, model='ridge_regression', poly_degrees=[2], inv_log_degrees=[2], max_iters=[10], gammas=[0.01], lambdas=[0.001], k_fold=10):
    k_indice = build_k_indices(y, k_fold)

    accs = []
    ws = []

    nb_grid = len(poly_degrees) * len(inv_log_degrees) * len(max_iters) * len(gammas) * len(lambdas)
    
    for k in range(k_fold):
        count = 1
        
        test_indice = k_indice[k]
        train_indice = k_indice[~(np.arange(k_indice.shape[0]) == k)]
        train_indice = train_indice.reshape(-1)

        ytest = y[test_indice]
        xtest = tx[test_indice]

        ytrain = y[train_indice]
        xtrain = tx[train_indice]

        xtest = xtest[:, ~(xtest.std(0) == 0)]
        xtrain = xtrain[:, ~(xtrain.std(0) == 0)]

        xtest = replaceNaNsWithMedian(xtest)
        xtrain = replaceNaNsWithMedian(xtrain)

        for poly_degree in poly_degrees:

            xtrain_poly_exp = degree_expansion(xtrain, poly_degree)
            xtest_poly_exp = degree_expansion(xtest, poly_degree)

            xtrain_stand, mean, std = standardize(xtrain_poly_exp[:,1:])
            xtest_stand = fit_standardize(xtest_poly_exp[:, 1:], mean, std)

            for log_degree in inv_log_degrees:

                xtrain_inv_log, valid_column = invert_log(xtrain, log_degree)
                xtest_inv_log = fit_invert_log(xtest, valid_column, log_degree)

                inv_log_stand, log_mean, log_std = standardize(xtrain_inv_log[:,1:])
                inv_log_stand_test = fit_standardize(xtest_inv_log[:,1:], log_mean, log_std)

                xtrain_process = np.column_stack((xtrain_stand, inv_log_stand))
                xtest_process = np.column_stack((xtest_stand, inv_log_stand_test))

                for max_iter in max_iters:
                    for gamma in gammas:
                        for lambda_ in lambdas:
                            print(f'Fold number: {str(k+1)}  {int(count/nb_grid*100)}%  ', end='\r', flush=True)
                            initial_w = init_w(xtrain_process)

                            w, acc = cross_validation(ytrain, xtrain_process, ytest, xtest_process, initial_w, max_iter, gamma, lambda_, model)

                            cross = {
                                'poly_degree': poly_degree,
                                'log_degree': log_degree,
                                'max_iteration': max_iter,
                                'gamma': gamma,
                                'lambda': lambda_,
                                #'weight': w,
                                'acc': acc
                            }

                            accs.append(cross)
                            ws.append(w)
                            
                            count += 1
    results = []
    for j in range(nb_grid):
        fold = [nb_grid * k + j for k in range(k_fold)]
        results.append(np.array(accs)[np.array(fold)])

    grid_results = []

    for grid in range(nb_grid):
        param_dict = {k: v for (k, v) in results[grid][0].items() if (k not in ['acc', 'weights'])}
        acc_mean = np.array([item['acc'] for item in results[grid]]).mean(0)
        acc_std = np.array([item['acc'] for item in results[grid]]).std(0)
        #weight_mean = np.array([item['weight'] for item in results[grid]]).mean(0)
        #weight_std = np.array([item['weight'] for item in results[grid]]).std(0)

        param_dict.update({'acc_mean': acc_mean, 
                           'acc_std': acc_std, 
                           #'weight_mean': weight_mean, 
                           #'weight_std': weight_std
                          })

        grid_results.append(param_dict)
    
    return grid_results
