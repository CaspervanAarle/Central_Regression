# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 09:59:43 2021

@author: Casper
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression


# data file location
dataset = "CASP_central"
    
#var_list = ["age", "bmi", "children"]
#target_list = ["charges"]

var_list = ["F1", "F2", "F3", "F4"]
target_list = ["RMSD"]

#var_list = ["doc-avail", "hosp_avail", "income_1000s", "pop_density"]
#target_list = ["death_rate_per_1000"]

test_split = 0
datapoints_amount = 150
standardize_data = True

def compare_coefficients1():
    """ comparing raw beta coefficients with converted coefficients """
    directory = "C:\\Users\\Casper\\Projects\\MasterScriptie\\custom_projects\\editing\\PHT_Preprocessing\\out\\{}\\data.csv".format(dataset)

    X = pd.read_csv(directory)[var_list].to_numpy()[:datapoints_amount]
    y = np.squeeze(pd.read_csv(directory)[target_list].to_numpy())[:datapoints_amount]
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    
    X_standardized = standardize(X, X_mean, X_std)
    
    model = LogisticRegression().fit(X, y)    
    model_standardized = LogisticRegression().fit(X_standardized, y)    
    
    print("coefficients ", model.coef_)
    print("beta coefficients ", model_standardized.coef_)
    
    for tuple_ in zip(model.coef_[0], X_std):
        standardized_coef = unstd_to_std_coef2_log(*tuple_)
        print(standardized_coef)
    
    for tuple_ in zip(model_standardized.coef_[0], X_std):
        unstd_coef = std_to_unstd_coef_log(*tuple_)
        print(unstd_coef)
        
    print("\nintercept ", model.intercept_)
    print("coef ", unstd_coef)
    print("xmean ", X_mean)
    
    
def compare_coefficients2():
    directory = "C:\\Users\\Casper\\Projects\\MasterScriptie\\custom_projects\\editing\\PHT_Preprocessing\\out\\{}\\data.csv".format(dataset)

    X = pd.read_csv(directory)[var_list].to_numpy()[:datapoints_amount]
    y = np.squeeze(pd.read_csv(directory)[target_list].to_numpy())[:datapoints_amount]
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    
    X_standardized = standardize(X, X_mean, X_std)
    y_standardized = standardize(y, y_mean, y_std)
    
    model = LinearRegression().fit(X, y)    
    model_standardized = LinearRegression().fit(X_standardized, y_standardized)    
    
    print("coefficients ", model.coef_)
    print("beta coefficients ", model_standardized.coef_)
    
    for tuple_ in zip(model.coef_, X_std, [y_std, y_std]):
        standardized_coef = unstd_to_std_coef_lin(*tuple_)
        print(standardized_coef)
        
    for tuple_ in zip(model_standardized.coef_, X_std, [y_std, y_std]):
        unstandardized_coef = std_to_unstd_coef_lin(*tuple_)
        print(unstandardized_coef)
        
        
    print("intercept = ", model.intercept_)
    intercept = y_mean - unstandardized_coef*X_mean[0]
    print(intercept)
    
def unstd_to_std_coef_lin(b, SD_x, SD_y):
    beta = b*(SD_x/SD_y)
    return beta

def std_to_unstd_coef_lin(beta, SD_x, SD_y):
    b = beta*(SD_y/SD_x)
    return b

def unstd_to_std_coef_log(b, SD_x):
    beta = np.exp(b)-1
    return beta

def unstd_to_std_coef2_log(b, SD_x):
    beta = b * SD_x
    return beta

def unstd_to_std_coef3_log(b, SD_x):
    beta = b * SD_x * np.pi/np.sqrt(3)
    return beta

def std_to_unstd_coef_log(beta, SD_x):
    b = beta / SD_x
    return b

def standardize(X, X_mean, X_std):
    return (X - X_mean) / X_std 

