# -*- coding: utf-8 -*-
"""
FILE NAME:      Modify_Run_Parameters.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    This script will correct and modify the .CSV simulation parameter 
                values.

INPUT
FILES(S):       1) .CSV file that stores all of the parameters used for a simulation.

OUTPUT
FILES(S):       1) .CSV file with updated parameters used for a simulation.


INPUT
ARGUMENT(S):    1) Main Input directory: The directory that houses all of the simulation
                data files.


CREATED:        01Jan23

MODIFICATIONS
LOG:            N/A

LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.8.8

VERSION:        1.0

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:    N/A

NOTE(S):        N/A

"""

import os, argparse
import numpy as np
import pandas as pd


def modify_parameters(input_directory):
    """
    This function searches through the entire directory system and if it 
    encounters a directory with .NPY files, specifically the one for the 
    filament position, it modifies the parameter CSV file and adds additional 
    information or modifies it.
    
    Inputs:
        
    input_directory:            The main parent directory that contains all
                                of the filament simulation data. 
    """
    for root,dirs,files in os.walk(input_directory):
        for dir_ in dirs:
            new_path = os.path.join(root,dir_)
            check_file_name = os.path.join(new_path,'filament_allstate.npy')
            if os.path.exists(check_file_name):
                param_csv_pth = os.path.join(new_path,'parameter_values.csv')
                param_csv = pd.read_csv(param_csv_pth,index_col = 0,header = 0)
                
                #Universal parameters
                param_csv.loc['Sterics Use','Value'] = 'Yes'
                param_csv.loc['Brownian Use','Value'] = 'Yes'
                
                #Flow-based parameters
                param_csv.loc['Flow Type','Value'] = 'Poiseuille'
                if param_csv.loc['Flow Type','Value'] == 'Kolmogorov':
                    param_csv.loc['Poiseuille U Centerline'] = 0
                    param_csv.loc['Kolmogorov Frequency','Value'] = 1
                    param_csv.loc['Kolmogorov Phase','Value'] = 'Pi'
                elif param_csv.loc['Flow Type','Value'] == 'Poiseuille':
                    new_channel_height = float(param_csv.loc['Channel Upper Height','Value'])
                    new_centerline = (1**2)/(0.5**2)*new_channel_height**2
                    param_csv.loc['Poiseuille U Centerline','Value'] = new_centerline
                    param_csv.loc['Kolmogorov Frequency','Value'] = 0
                    param_csv.loc['Kolmogorov Phase','Value'] = 0
                elif param_csv.loc['Flow Type','Value'] == 'Shear':
                    param_csv.loc['Poiseuille U Centerline'] = 0
                    param_csv.loc['Kolmogorov Frequency','Value'] = 0
                    param_csv.loc['Kolmogorov Phase','Value'] = 0
                
                param_csv.to_csv(param_csv_pth)
    # return param_csv_pth,param_csv
                

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_directory", 
                        help="The main directory where the simulation files reside in",
                    type = str)
    args = parser.parse_args(['C:/Users/super/OneDrive - University of California, Davis/School/UCD_Files/Work/00_Projects/02_Shear_Migration/00_Scripts/01_Migration_Simulations/01_Test_Results/Poiseuille_Flow/']) #Uncommment this line when debugging
    # args = parser.parse_args()
    
    # param_csv_pth,param_csv = modify_parameters(args.input_directory)
    modify_parameters(args.input_directory)
                
                
