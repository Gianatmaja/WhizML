#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python script contains the codes that will perform the model saving process. It is
one of the pipelines in the WhizML project.

For more information, visit the WhizML GitHub repository: 
"""

# Import required modules
import numpy as np
import pandas as pd
import wandb
import yaml
import subprocess
import pickle

# Import required classes
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from eda import AutoEDA


# Define ModelSaving class
class ModelSaving:
    
    # Define saveModel function
    def saveModel(self, df_train, df_test, target, model_url):
        # This model will obtain the chosen model's configurations from Weights & Biases,
        # and will proceed to train that model on the full training dataset.
        
        api = wandb.Api()
    
        # Best acc model details
        run = api.run(model_url)
        config = run.config
    
        # Get X and y
        y = df_train[target]
        X = df_train.drop(target, axis = 1)
    
        # Train best model with full training data
        if conf['problem']['classification']['tag'] == True:
            
            if config['model_name'] == 'Random Forest':
                best = RandomForestClassifier(max_depth = config['max_depth'], n_estimators = config['n_estimators'], random_state = 1)
                best.fit(X, y)
            elif config['model_name'] == 'XGBoost':
                best = XGBClassifier(max_depth = config['max_depth'], n_estimators = config['n_estimators'], random_state = 1)
                best.fit(X, y)
            else:
                best = LogisticRegression(penalty = config['penalty'], C = config['C'], random_state = 1)
                best.fit(X, y)
                
        else:
            
            if config['model_name'] == 'Random Forest':
                best = RandomForestRegressor(max_depth = config['max_depth'], n_estimators = config['n_estimators'], random_state = 1)
                best.fit(X, y)
            elif config['model_name'] == 'XGBoost':
                best = XGBRegressor(max_depth = config['max_depth'], n_estimators = config['n_estimators'], random_state = 1)
                best.fit(X, y)
            else:
                best = Ridge(C = config['C'], random_state = 1)
                best.fit(X, y)
                
        # Get predictions
        y_test = df_test[target]
        X_test = df_test.drop(target, axis = 1)
        
        preds = best.predict(X_test)
        X_test['True'] = y_test
        X_test['Predicted'] = preds
        
        return best, X_test


if __name__ == '__main__':
    
    # Initialize the AutoEDA and ModelSaving classes
    prep_launcher = AutoEDA()
    model_launcher = ModelSaving()
    
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
    model_url = conf['wandb']['best_model_url']
    model_path = conf['other_directories']['model'] + '/model.pkl'
    
    # Get data
    df_train = prep_launcher.getData(data_path_train)
    df_test = prep_launcher.getData(data_path_test)
    
    # Get model
    best_model, predictions = model_launcher.saveModel(df_train, df_test, target, model_url)
    pickle.dump(best_model, open(model_path, 'wb'))
    
    # Print status for user
    current_time = datetime.now()
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    print(current_time_str, '\033[93mSTATUS\033[0m - Model saved to data/model/ directory.')
    
    # Save predictions
    preds_path = conf['other_directories']['predictions'] + '/preds.csv'
    predictions.to_csv(preds_path, index = False)
    
    # Print status for user
    current_time = datetime.now()
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    print(current_time_str, '\033[93mSTATUS\033[0m - Predictions saved to data/model_output/ directory.')
    
    
