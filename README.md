# CS-433 Machine Learning - Project 1
This project for the Machine Learning course is about predicting from a data set whether a particle is a Higgs Boson based on thirty features.

All the following functions have to be executed in an environment with ``numpy`` installed.

The content of this project is composed of different parts:
- The folder ``data`` contains the data for the project. It is composed of three files:
   - ``train.csv``: data set used for training the model.
   - ``test.csv``: data set used to make prediction from the model.
   - ``sample-submission.csv``: format in which the predictions are presented for submission.
   
- The folder ``report`` contains:
   - ``report.pdf``: report in PDF format.
   
- The folder ``ML2019Project1`` containing the python code:
   - __``project1.ipynb``__ : notebook containing the procedure for the model selection.
   - __``preprocess.py``__: contains methods for preprocessing the data.
        -``standardize``: standardizes the data.
         
        -``fit_standardize``: standardizes the data based on *given* mean and standard deviation. 
         
        -``remove_constant_col``: removes the data features with *zero* standard deviation.
         
        -``fit_remove_constant_col``: removes data features based on *given* features.
         
        -``get_categorical_masks``: returns four masks corresponding to the categorical values 0,1,2 and 3 respectively.
         
        -``replaceNaNsWithMedian``: replaces the *-999.0* values with the median of the corresponding feature.
         
   - __``proj_helpers.py``__: predefined help functions for the project.
        -``load_csv_data``:loads the data from CSV.
        
        -``predict_labels``: predicts label based on data matrix and weight vector.
        
        -``create_csv_submission``:creates CSV files for submission.
         
   - __``feature_engineer.py``__ : contains features engineering functions.
        -``degree_expansion``: builds polynomial expansion features for a feature.
        
        -``invert_log``: builds inverse log values with degree expansion passed in arguments which are positive in value.
        
        -``fit_invert_log``: builds inverse log values with degree expansion with predefined features.
         
   - __``helpers.py``__ : contains necessary functions to build our model.
        -``compute_loss``: computes the loss function using the MSE.
        
        -``compute_gradient``: computes the gradient of the loss function using MSE.
        
        -``batch_iter``: generates a minibatch iterator for dataset.
        
        -``init_w``: initializes the weight vector.
        
        -``accuracy``: compute the accuracy of predictors.
        
        -``sigmoid``: sigmoid function.
        
        -``compute_gradient_logistic``: computes the gradient of the logistic loss.
        
        -``compute_logistic_loss``: computes the logistic regression loss.
        
        -``find_best_parameters``: extract the model's parameter with the best accuracy.
        
        -``accuracy_vizualization1``:
        
        -``accuracy_vizualization2``:
         
   - __``implementations.py``__ : contains all regression models for the project.
        -``least_square_GD``: linear regression using gradient descent.
        
        -``least_square_SGD``: linear regression using stochastic gradient descent.
        
        -``least_square``: linear regression using normal equations.
        
        -``ridge_regression``: ridge regression using normal equations.
        
        -``logistic_regression``: logistic regression using stochastic gradient descent.
        
        -``reg_logistic_regression``: regularized logistic regression using stochastic gradient descent.
         
   - __``cross_validation.py``__: contains all the necessary methods to perform a cross validation.
        -``build_k_indices``: builds k indices for k-fold cross validation.
        
        -``cross_validation``: performs cross validation based on a given model.
        
        -``cross_validation_demo``: call 'cross_validation' with ranges of parameters to get the best combination.
         
   - __``run-py``__: contains the procedure that generates the exact CSV file for submission.
   
