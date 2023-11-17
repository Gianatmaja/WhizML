#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python script contains the codes that will perform the data drift analysis process. It is
one of the pipelines in the WhizML project.

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
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

from eda import AutoEDA


# Define DriftDetector class
class DriftDetector:

    # Define detectDrift function
    def detectDrift(self, data1, data2, report_dir):
        # This function will compare 2 datasets to check for data drift
        
        # Define report
        report = Report(metrics = [
            DataDriftPreset(), 
        ])
        
        report_filepath = report_dir + '/report.html'
        
        # Obtain and save drift analysis report
        report.run(reference_data = data1, current_data = data2)
        report.save_html(report_filepath)
        
        # Print status for user
        current_time = datetime.now()
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        print(current_time_str, '\033[93mSTATUS\033[0m - Saving report to data/reporting/ directory.')
        
        
        return None


if __name__ == '__main__':
    
    # Initialize the AutoEDA and DriftDetector classes
    prep_launcher = AutoEDA()
    drift_launcher = DriftDetector()
    
    # Read config file
    conf = yaml.safe_load(Path('config.yml').read_text())
    
    # Read in dataset
    data_path_1 = conf['data_path']['additional_data1']
    data_path_2 = conf['data_path']['additional_data2']
    
    # Other required inputs
    report_dir = conf['other_directories']['reporting']

    # Get data
    df1 = prep_launcher.getData(data_path_1)
    df2 = prep_launcher.getData(data_path_2)
    
    # Detect for drift
    drift_launcher.detectDrift(df1, df2, report_dir)
    
    # Print status for user
    current_time = datetime.now()
    current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    print(current_time_str, '\033[93mSTATUS\033[0m - Process completed.')
    
    
    