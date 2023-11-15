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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from eda import AutoEDA


# Define ModelSaving class
class ModelSaving:
    
    # Define saveModel function
    def saveModel(self, df, target, model_url):
        # This model will obtain the chosen model's configurations from Weights & Biases,
        # and will proceed to train that model on the full training dataset.
        
        api = wandb.Api()
    
        # Best acc model details
        run = api.run(model_url)
        config = run.config
    
        # Get X and y
        y = df[target]
        X = df.drop(target, axis = 1)
    
        # Train best model with full training data
        if config['model_name'] == 'Random Forest':
            best = RandomForestClassifier(max_depth = config['max_depth'], n_estimators = config['n_estimators'], random_state = 1)
            best.fit(X, y)
        else:
            best = XGBClassifier(max_depth = config['max_depth'], n_estimators = config['n_estimators'], random_state = 1)
            best.fit(X, y)

        return best


if __name__ == '__main__':
    
    # Initialize the AutoEDA and ModelSaving classes
    prep_launcher = AutoEDA()
    model_launcher = ModelSaving()
    
    # Read config file
    conf = yaml.safe_load(Path('config.yml').read_text())
    
    # Read in dataset
    data_path_train = conf['data_path']['model_input'] + '/train.csv'
    
    # Other required inputs
    target = conf['problem']['classification']['target']
    model_url = conf['wandb']['best_model_url']
    model_path = conf['other_directories']['model'] + '/model.pkl'
    
    # Get data
    df_train = prep_launcher.getData(data_path_train)
    
    # Get model
    best_model = model_launcher.saveModel(df_train, target, model_url)
    pickle.dump(best_model, open(model_path, 'wb'))
    
    # Print status for user
    current_time = datetime.now()
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    print(current_time_str, '\033[93mSTATUS\033[0m - Model saved to data/model/ directory.')
    
    
    