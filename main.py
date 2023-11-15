#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:26:13 2023

@author: gianatmaja
"""

import yaml
import subprocess

from pathlib import Path

class MasterLauncher:
    
    def launchProcess(self, master_config):
        
        if (master_config['task']['eda'] == True):
            script_path = 'src/pipeline/eda.py'
            subprocess.run(['python', script_path])
            
        if (master_config['task']['model_experimentation'] == True):
            script_path = 'src/pipeline/model_experimentation.py'
            subprocess.run(['python', script_path])

        if (master_config['task']['model_finalization'] == True):
            script_path = 'src/pipeline/model_saving.py'
            subprocess.run(['python', script_path])

        if (master_config['task']['model_explainability'] == True):
            script_path = 'src/pipeline/model_explanation.py'
            subprocess.run(['python', script_path])
            
        if (master_config['task']['data_drift_analysis'] == True):
            script_path = 'src/pipeline/data_drift.py'
            subprocess.run(['python', script_path])
            
        if (master_config['task']['bias_analysis_data_prep'] == True):
            script_path = 'src/pipeline/bias_analysis_prep.py'
            subprocess.run(['python', script_path])
        
        return None

if __name__ == '__main__':
    # Launch launcher
    master_launcher = MasterLauncher()
    
    # Get config file
    master_config = yaml.safe_load(Path('config.yml').read_text())
    
    # Execute
    master_launcher.launchProcess(master_config)

