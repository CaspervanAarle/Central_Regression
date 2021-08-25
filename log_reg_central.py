# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 11:13:05 2021

@author: Casper
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from common import build_save_file, standardize
    
# variables
#readmission hospital
#dataset = "readmission_hospital_central"
#var_list = ["time_in_hospital",	"num_lab_procedures",	"num_procedures", "num_medications"]#	"num_medications",	"number_outpatient",	"number_emergency",	"number_inpatient",	"number_diagnoses"]
#target_list = ["readmitted"]
#coimbra

#dataset = "haberman_central"
#var_list =  ["age",	"operation_year",	"nodes_detected"]	
#target_list =  ["died_within_5y"]

#dataset = "cardio_train_continuous_central"
#var_list =  ["weight"]
#target_list =  ["cardio"]

dataset = "cardio_train_binary_central"
var_list =  ["gender",	"smoke",	"alco"]
target_list =  ["cardio"]

test_split = 0
datapoints_amount = 150
standardize_data = False
    
def start():
    """ Main training loop"""
    ### load data:
    directory = "C:\\Users\\Casper\\Projects\\MasterScriptie\\custom_projects\\editing\\PHT_Preprocessing\\out\\{}\\data.csv".format(dataset)
    X = pd.read_csv(directory)[var_list].to_numpy()[:datapoints_amount]
    y = np.squeeze(pd.read_csv(directory)[target_list].to_numpy())[:datapoints_amount]
    
    ### standardize data:
    if standardize_data is True:
        print("data standardized")
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = standardize(X, X_mean, X_std)
        
    ### subset training?    
    if(test_split == 0):
        model = train_model(X, y, [], [])
    else:
        X_train, X_test = train_test_split(X, test_size=test_split, random_state=1)
        y_train, y_test = train_test_split(y, test_size=test_split, random_state=1)
        # standardize on train set statistics:
        X_train_standardized = standardize(X_train, np.mean(X_train, axis=0), np.std(X_train, axis=0))
        X_test_standardized = standardize(X_test, np.mean(X_train, axis=0), np.std(X_train, axis=0))
            
        model = train_model(X_train_standardized, y_train, X_test_standardized, y_test)

    ### save model
    if standardize_data is True:
        build_save_file(model, directory, standardize_data, var_list, target_list, X, X_mean, X_std)
    else:
        build_save_file(model, directory, standardize_data, var_list, target_list, X)
    


def train_model(X, y, X_test, y_test):
    """ Trains a model and visualizes this model when input dimension is 1"""
    # train model
    model = LogisticRegression(max_iter=1000000, penalty='none').fit(X, y)
    
    # visualize if possible
    if(len(var_list) == 1):
        # visualize model:
        X_test_lin = np.linspace(min(X)-2, max(X)+2, 300)
        plt.scatter(X_test_lin,model.predict_proba(X_test_lin)[:,1])
        plt.scatter(X, y)
        # visualize datapoints:
        plt.scatter(X_test, y_test)
        plt.show()
    return model

if __name__ == "__main__" :
    start()
        
    
    
    
    
    
    
    