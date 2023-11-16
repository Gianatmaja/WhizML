#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python script contains the codes that will perform the model experimentation process. It is
one of the pipelines in the WhizML project.

For more information, visit the WhizML GitHub repository: 
"""

# Import required modules
import numpy as np
import pandas as pd
import wandb
import yaml
import subprocess

# Import required classes
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from eda import AutoEDA

# Define the Experiments class
class Experiments:
    
    # Define the trainRfModels function
    def trainRfModels(self, df_train, df_test, target, project_name, job_type):
        # This function will train several configurations of the Random-Forest algorithm,
        # and will record the results to Weights & Biases.
        
        # Get data
        y_train = df_train[target]
        X_train = df_train.drop(target, axis = 1)
        
        y_test = df_test[target]
        X_test = df_test.drop(target, axis = 1)
        
        # Define hyperparameter values to try
        grid_values = {'n_estimators': [200, 300, 400], 'max_depth': [2, 3, 4, 5]}

        # Define algorithm
        rf = RandomForestClassifier(max_features = int(X_train.shape[1]/3), random_state = 1)
        
        # Iterate through hyperparameter combinations
        for params in ParameterGrid(grid_values):
            # Create a new run for each combination
            with wandb.init(project = project_name, job_type = job_type, config = params):
                wandb.run.config["model_name"] = "Random Forest"
                # Train a Random Forest model with the current hyperparameters
                rf.set_params(**params)
                rf.fit(X_train, y_train)
        
                # Predict on the test set
                y_pred_rf = rf.predict(X_test)
        
                # Calculate and log metrics
                classification_rep = classification_report(y_test, y_pred_rf, output_dict = True)
                wandb.run.summary.update(params)
                wandb.run.summary.update(classification_rep)
        
        # Finish the WandB run
        wandb.finish()
        
        # Print status for user
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(current_time_str, '\033[93mSTATUS\033[0m - Random forest model training concluded.')
        
        return None

    
    def trainXgbModels(self, df_train, df_test, target, project_name, job_type):
        # This function will train several configurations of the XGBoost algorithm,
        # and will record the results to Weights & Biases.
        
        # Get data
        y_train = df_train[target]
        X_train = df_train.drop(target, axis = 1)
        
        y_test = df_test[target]
        X_test = df_test.drop(target, axis = 1)
        
        # Define hyperparameter values to try
        grid_values = {'n_estimators': [100, 300, 500], 'max_depth': [3, 5, 7, 10]}

        # Define algorithm
        xgb = XGBClassifier(random_state = 1)
        
        # Iterate through hyperparameter combinations
        for params in ParameterGrid(grid_values):
            # Create a new run for each combination
            with wandb.init(project = project_name, job_type = job_type, config = params):
                wandb.run.config["model_name"] = "XGBoost"
                # Train an XGBoost model with the current hyperparameters
                xgb.set_params(**params)
                xgb.fit(X_train, y_train)
        
                # Predict on the test set
                y_pred_xgb = xgb.predict(X_test)
        
                # Calculate and log metrics
                classification_rep = classification_report(y_test, y_pred_xgb, output_dict = True)
                wandb.run.summary.update(params)
                wandb.run.summary.update(classification_rep)
        
        # Finish the WandB run
        wandb.finish()
        
        # Print status for user
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(current_time_str, '\033[93mSTATUS\033[0m - XGBoost model training concluded.')
        
        return None
    

    def trainLogRModels(self, df_train, df_test, target, project_name, job_type):
        # This function will train several configurations of the Logistic Regression algorithm,
        # and will record the results to Weights & Biases.
        
        # Get data
        y_train = df_train[target]
        X_train = df_train.drop(target, axis = 1)
        
        y_test = df_test[target]
        X_test = df_test.drop(target, axis = 1)
        
        # Define hyperparameter values to try
        grid_values = {'C': np.logspace(-4, 4, 9)}

        # Define algorithm
        lr = LogisticRegression(penalty = 'l2', random_state = 1)
        
        # Iterate through hyperparameter combinations
        for params in ParameterGrid(grid_values):
            # Create a new run for each combination
            with wandb.init(project = project_name, job_type = job_type, config = params):
                wandb.run.config["model_name"] = "Logistic Regression"
                # Train a Logistic Regression model with the current hyperparameters
                lr.set_params(**params)
                lr.fit(X_train, y_train)
        
                # Predict on the test set
                y_pred_lr = lr.predict(X_test)
        
                # Calculate and log metrics
                classification_rep = classification_report(y_test, y_pred_lr, output_dict = True)
                wandb.run.summary.update(params)
                wandb.run.summary.update(classification_rep)
        
        # Finish the WandB run
        wandb.finish()
        
        # Print status for user
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(current_time_str, '\033[93mSTATUS\033[0m - Logistic regression model training concluded.')
        
        return None


    def trainLinRModels(self, df_train, df_test, target, project_name, job_type):
        # This function will train several configurations of the Linear Regression algorithm,
        # and will record the results to Weights & Biases.
        
        # Get data
        y_train = df_train[target]
        X_train = df_train.drop(target, axis = 1)
        
        y_test = df_test[target]
        X_test = df_test.drop(target, axis = 1)
        
        # Define hyperparameter values to try
        grid_values = {'alpha': [0.01, 0.1, 1, 10]}

        # Define algorithm
        lr = Ridge(random_state = 1)
        
        # Iterate through hyperparameter combinations
        for params in ParameterGrid(grid_values):
            # Create a new run for each combination
            with wandb.init(project = project_name, job_type = job_type, config = params):
                wandb.run.config["model_name"] = "Ridge Regression"
                # Train a Linear Regression model with the current hyperparameters
                lr.set_params(**params)
                lr.fit(X_train, y_train)
        
                # Predict on the test set
                y_pred_lr = lr.predict(X_test)
        
                # Calculate and log metrics
                r2 = r2_score(y_test, y_pred_lr)
                mae = mean_absolute_error(y_test, y_pred_lr)
                rmse = mean_squared_error(y_test, y_pred_lr, squared = False)
                
                wandb.run.summary.update(params)
                wandb.run.summary.update({'r_squared': r2})
                wandb.run.summary.update({'mae': mae})
                wandb.run.summary.update({'rmse': rmse})
        
        # Finish the WandB run
        wandb.finish()
        
        # Print status for user
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(current_time_str, '\033[93mSTATUS\033[0m - Linear regression model training concluded.')
        
        return None
    
    
    def trainRfRModels(self, df_train, df_test, target, project_name, job_type):
        # This function will train several configurations of the Random-Forest algorithm,
        # and will record the results to Weights & Biases.
        
        # Get data
        y_train = df_train[target]
        X_train = df_train.drop(target, axis = 1)
        
        y_test = df_test[target]
        X_test = df_test.drop(target, axis = 1)
        
        # Define hyperparameter values to try
        grid_values = {'n_estimators': [200, 300, 400], 'max_depth': [2, 3, 4, 5]}

        # Define algorithm
        rf = RandomForestRegressor(max_features = int(X_train.shape[1]/3), random_state = 1)
        
        # Iterate through hyperparameter combinations
        for params in ParameterGrid(grid_values):
            # Create a new run for each combination
            with wandb.init(project = project_name, job_type = job_type, config = params):
                wandb.run.config["model_name"] = "Random Forest"
                # Train a Random Forest model with the current hyperparameters
                rf.set_params(**params)
                rf.fit(X_train, y_train)
        
                # Predict on the test set
                y_pred_rf = rf.predict(X_test)
        
                # Calculate and log metrics
                r2 = r2_score(y_test, y_pred_rf)
                mae = mean_absolute_error(y_test, y_pred_rf)
                rmse = mean_squared_error(y_test, y_pred_rf, squared = False)
                
                wandb.run.summary.update(params)
                wandb.run.summary.update({'r_squared': r2})
                wandb.run.summary.update({'mae': mae})
                wandb.run.summary.update({'rmse': rmse})
        
        # Finish the WandB run
        wandb.finish()
        
        # Print status for user
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(current_time_str, '\033[93mSTATUS\033[0m - Random forest model training concluded.')
        
        return None
    
    
    def trainXgbRModels(self, df_train, df_test, target, project_name, job_type):
        # This function will train several configurations of the XGBoost algorithm,
        # and will record the results to Weights & Biases.
        
        # Get data
        y_train = df_train[target]
        X_train = df_train.drop(target, axis = 1)
        
        y_test = df_test[target]
        X_test = df_test.drop(target, axis = 1)
        
        # Define hyperparameter values to try
        grid_values = {'n_estimators': [100, 300, 500], 'max_depth': [3, 5, 7, 10]}

        # Define algorithm
        xgb = XGBRegressor(random_state = 1)
        
        # Iterate through hyperparameter combinations
        for params in ParameterGrid(grid_values):
            # Create a new run for each combination
            with wandb.init(project = project_name, job_type = job_type, config = params):
                wandb.run.config["model_name"] = "XGBoost"
                # Train an XGBoost model with the current hyperparameters
                xgb.set_params(**params)
                xgb.fit(X_train, y_train)
        
                # Predict on the test set
                y_pred_xgb = xgb.predict(X_test)
        
                # Calculate and log metrics
                r2 = r2_score(y_test, y_pred_xgb)
                mae = mean_absolute_error(y_test, y_pred_xgb)
                rmse = mean_squared_error(y_test, y_pred_xgb, squared = False)
                
                wandb.run.summary.update(params)
                wandb.run.summary.update({'r_squared': r2})
                wandb.run.summary.update({'mae': mae})
                wandb.run.summary.update({'rmse': rmse})
        
        
        # Finish the WandB run
        wandb.finish()
        
        # Print status for user
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(current_time_str, '\033[93mSTATUS\033[0m - XGBoost model training concluded.')
        
        return None



if __name__ == '__main__':
    
    # Initialize the AutoEDA & Experiments classes
    prep_launcher = AutoEDA()
    ml_launcher = Experiments()
    
    # Read config file
    conf = yaml.safe_load(Path('config.yml').read_text())
    
    # Read in dataset
    data_path_train = conf['data_path']['model_input'] + '/train.csv'
    data_path_test = conf['data_path']['model_input'] + '/test.csv'
    
    # Other required inputs
    if conf['problem']['classification']['tag'] == True:
        target = conf['problem']['classification']['target']
        
    if conf['problem']['regression']['tag'] == True:
        target = conf['problem']['regression']['target']
        
    project_name = conf['wandb']['project']
    
    # Get data
    df_train = prep_launcher.getData(data_path_train)
    df_test = prep_launcher.getData(data_path_test)
    
    # Train models
    if conf['problem']['classification']['tag'] == True:
        
        if conf['problem']['classification']['binary'] == True:
            
            ml_launcher.trainLogRModels(df_train, df_test, target, project_name, job_type = 'hyperparameter_tuning_logr')
            
        ml_launcher.trainRfModels(df_train, df_test, target, project_name, job_type = 'hyperparameter_tuning_rf')
        ml_launcher.trainXgbModels(df_train, df_test, target, project_name, job_type = 'hyperparameter_tuning_xgb')
    
    
    if conf['problem']['regression']['tag'] == True:
        
        ml_launcher.trainLinRModels(df_train, df_test, target, project_name, job_type = 'hyperparameter_tuning_lr')
        ml_launcher.trainRfRModels(df_train, df_test, target, project_name, job_type = 'hyperparameter_tuning_rf')
        ml_launcher.trainXgbRModels(df_train, df_test, target, project_name, job_type = 'hyperparameter_tuning_xgb')
    
    
    
    
    