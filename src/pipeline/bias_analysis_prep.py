#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python script contains the codes that will perform the data preparation process for
bias analysis. It is one of the pipelines in the WhizML project.

For more information, visit the WhizML GitHub repository: 
"""

# Import required modules
import numpy as np
import pandas as pd
import yaml
import subprocess

# Import required classes
from pathlib import Path
from datetime import datetime

from eda import AutoEDA


# Define BiasDataPrep class
class BiasDataPrep:
    
    # Define the prepareDataset function
    def prepareDataset(self, df):
        # This function will prepare a dataset that can be used to analyse and detect
        # potential biases in the chosen model. It makes use of the Aequitas web app.
        
        new_order = ['Predicted', 'True'] + [col for col in df.columns if col != 'True' and col != 'Predicted']
        df = df[new_order]
        
        new_names = ['score', 'label_value'] + [col for col in df.columns if col != 'True' and col != 'Predicted']
        df.columns = new_names
        
        return df
    
if __name__ == '__main__':
    
    # Initialize the AutoEDA and BiasDataPrep classes
    prep_launcher = AutoEDA()
    bias_data_launcher = BiasDataPrep()
    
    # Read config file
    conf = yaml.safe_load(Path('config.yml').read_text())
    
    # Required configs
    data_path = conf['other_directories']['predictions'] + '/preds.csv'
    output_path = conf['other_directories']['reporting'] + '/output.csv'
    
    # Get data
    df = prep_launcher.getData(data_path)
    
    # Run the process for binary classification only
    if conf['problem']['classification']['binary'] == True:
        df_out = bias_data_launcher.prepareDataset(df)
        df_out.to_csv(output_path, index = False)
        
        # Print status for user
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(current_time_str, '\033[93mSTATUS\033[0m - Data saved to data/reporting/ directory.')
    
    else:
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(current_time_str, '\033[91mFAILED\033[0m - Bias analysis is only available for binary classification problems.')