# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 10:04:56 2021

@author: Casper
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import ShuffleSplit
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import random
from random import randint
import pickle
import json

def standardize(X, X_mean, X_std):
    return (X - X_mean) / X_std 

def save_model(name, scikit_model, *hyperparams):
    save_dir = "results/"
    # save the model to disk
    pickle.dump(scikit_model, open(save_dir + name + '.sav', 'wb'))
    
    # save hyperparameters to disk
    with open(save_dir + name + '.json','w') as jsonFile:
        json.dump(hyperparams[0], jsonFile)

def build_save_file(model, directory, standardize_data, var_list, target_list, X, X_mean=None, X_std=None, y_mean=None, y_std=None):    
    # save model
    model_name = "central_linear_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(model_name)
    if standardize_data is True:
        print("standardized data saved")
        save_model(model_name, model, {
                                            "data_dir": directory,
                                            "n_nodes":X.shape[0],
                                            "input_vars": var_list ,
                                            "target_vars": target_list ,
                                            "mu_list":X_mean.tolist() + [y_mean],
                                            "sigma_list":X_std.tolist()+ [y_std]
                                            })
    else:
        print("original data saved")
        save_model(model_name, model, {
                                            "data_dir": directory,
                                            "n_nodes": X.shape[0],
                                            "input_vars": var_list ,
                                            "target_vars": target_list}   )
    print("model learned and saved")