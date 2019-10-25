# CS-433 Machine Learning - Project 1
This project for the Machine Learning course is about predicting from a data set whether a particle is a Higgs Boson based on thirty features.

The content of this project is composed of different parts:
- The folder ``data`` contains the data for the project. It is composed of three files:
   - ``train.csv``: data set used for training the model.
   - ``test.csv``: data set used to make prediction from the model.
   - ``sample-submission.csv``: format in which the predictions are presented.
   
- The ``.py``files:
   - ``__preprocess.py__``: contains methods for preprocessing the data.
         -``standardize``: standardizes the data.
         -``fit_standardize``: standardizes the data based on *given* mean and standard deviation. 
   
