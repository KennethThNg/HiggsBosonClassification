import numpy as np

def degree_expansion(x, degree):
    poly_exp = np.ones(x.shape[0])
    for d in range(1, degree+1):
        poly_exp = np.column_stack((poly_exp, x**d))

    return poly_exp

def invert_log(x, degree):
    valid_column = (x.min(0) >= 0)
    x_invert_log = np.log(1/(1 + x[:,valid_column]))
    invert_log_poly = degree_expansion(x_invert_log, degree)
    return invert_log_poly, valid_column

def fit_invert_log(x, valid_column, degree):#for the test
    assert (x[:, valid_column.min(0) < 0]).sum() == 0
    x_inv_log = np.log(1/(1 + x[:,valid_column]))
    inv_log_poly = degree_expansion(x_inv_log, degree)
    return inv_log_poly
