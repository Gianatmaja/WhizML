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
    def prepareDataset(df, target):
        # This function will prepare a dataset that can be used to analyse and detect
        # potential biases in the chosen model. It makes use of the Aequitas web app.
        
        pass
    
if __name__ == '__main__':
    
    # Initialize the AutoEDA and BiasDataPrep classes
    prep_launcher = AutoEDA()
    bias_data_launcher = BiasDataPrep()
    
    # Read config file
    conf = yaml.safe_load(Path('config.yml').read_text())
    
    # Run the process for binary classification only
    if conf['problem']['classification']['binary'] == True:
        pass
    
    else:
        print('Bias analysis is only available for binary classification problems.')
        