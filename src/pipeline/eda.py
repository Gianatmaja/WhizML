#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python script contains the codes that will perform the data reading and auto-EDA process. It is
one of the pipelines in the WhizML project.

For more information, visit the WhizML GitHub repository: 
"""

# Import required modules
import numpy as np
import pandas as pd
import dtale
import yaml

# Import required classes
from pathlib import Path
from datetime import datetime


# Define AutoEDA class
class AutoEDA:
    
    # Define getData function
    def getData(self, path):
        # This function will read in a csv dataset
        
        df = pd.read_csv(path)
        
        return df
    
    # Define launchAutoEDA function
    def launchAutoEDA(self, df):
        # This function will start the d-tale auto-EDA process
        
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(current_time_str, '\033[91mIMPORTANT\033[0m    - Press Ctrl + C to exit the Auto-EDA process.')
        
        d = dtale.show(df, subprocess = False)
        
        return None

if __name__ == '__main__':
    # Initialize AutoEDA class
    launcher = AutoEDA()
    
    # Read config file
    conf = yaml.safe_load(Path('config.yml').read_text())
    
    # Read in dataset
    data_path = conf['data_path']['raw']
    df = launcher.getData(data_path)
    
    # Launch D-Tale auto EDA dashboard
    launcher.launchAutoEDA(df)