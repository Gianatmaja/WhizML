#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python script contains the codes that will perform the model explanation process. It is
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
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

from eda import AutoEDA


# Define ModelExplainer class
class ModelExplainer:
    
    # Define explainModel function
    def explainModel(self, model, data, target):
        # This function will run the explainer dashboard, which will offer
        # insights to explain the model used in the project
        
        y = data[target]
        X = data.drop(target, axis = 1)

        explainer = ClassifierExplainer(model, X, y)
        ExplainerDashboard(explainer).run()
        
        return None
    

if __name__ == '__main__':
    
    # Intialize the AutoEDA and ModelExplainer classes
    prep_launcher = AutoEDA()
    exp_launcher = ModelExplainer()
    
    # Read config file
    conf = yaml.safe_load(Path('config.yml').read_text())
    
    # Read in dataset
    data_path_test = conf['data_path']['model_input'] + '/test.csv'
    
    # Other required inputs
    target = conf['problem']['classification']['target']
    model_path = conf['other_directories']['model'] + '/model.pkl'

    # Get data
    df_test = prep_launcher.getData(data_path_test)
    
    # Get model
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    
    # Get model explanation
    exp_launcher.explainModel(loaded_model, df_test, target)
    
    
