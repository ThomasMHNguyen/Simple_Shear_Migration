# -*- coding: utf-8 -*-
"""
FILE NAME:      Compile_Simulation_Parameters.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    This script compile all of the parameter values from successful
                simulations as well as unsuccessful simulations.

INPUT
FILES(S):       1) .CSV file that stores all of the parameters used for a simulation.

OUTPUT
FILES(S):       1) .CSV file that contains all relevant simulation parameters.


INPUT
ARGUMENT(S):    1) Main directory: The directory that houses ALL of the simulation
                data files.


CREATED:        02Feb23

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
import re, sys, os, argparse, shutil, glob
import numpy as np
import pandas as pd

class simulation_params():
    """
    This class compiles all of the simulation parameters that both failed and
    suceeded into a single CSV file.
    """
    def __init__(self,input_directory,output_directory):
        """
        This function initializes the class. 
        
        Inputs:
            
        input_directory:            Absolute path to the simulation directory that 
                                    contains all of the simulation data.
        input_directory:            Absolute path to the directory that 
                                    you want the output CSV file to be written to.                       
        """
        self.input_dir = input_directory
        self.output_dir = output_directory
        self.all_params_list = []
        
    def read_files(self):
        """
        This function walks through the entire contents of the input directory
        to read the CSV files corresponding to successful simulation runs and 
        unsuccessful simulation runs.        
        """
        for root,dirs,files in os.walk(self.input_dir):
            for dir_ in dirs:
                self.new_path = os.path.join(root,dir_)
                self.check_file_name = os.path.join(self.new_path,'filament_allstate.npy')
                if os.path.exists(self.check_file_name):
                    self.valid_param_path = os.path.join(self.new_path,'parameter_values.csv')
                    self.fail_param_path = glob.glob(os.path.join(self.new_path,'failed_run_*.csv'))
                    if os.path.exists(self.valid_param_path):
                        self.all_params_list.append(
                            self.read_valid_simulation_data(
                            self.valid_param_path,file_type = 'valid'))
                    if self.fail_param_path:
                        for file in self.fail_param_path:
                            self.all_params_list.append(
                                self.read_valid_simulation_data(
                                    file,file_type = 'fail'))
        
        self.all_simul_params = pd.concat(
            [pd.DataFrame.from_dict(i) for i in self.all_params_list],ignore_index = True)
        self.all_simul_params.to_csv(os.path.join(self.output_dir,'all_simulation_parameters.csv'))
    def read_valid_simulation_data(self,param_csv_path,file_type):
        """
        This function will read the parameter CSV file and save the relevant fields.
        
        Inputs:
        
        param_csv_path:             Absolute path to the parameter CSV file.
        file_type:                  String that lists whether the parameter CSV
                                    file corresponded to a sucessful or failed run.
        """
    
        self.param_csv  = pd.read_csv(param_csv_path,index_col = 0,header = 0)
        self.random_seed = float(self.param_csv.loc['Random Seed','Value'])
        self.true_dt = float(self.param_csv.loc['True dt','Value'])
        self.adpt_dt = float(self.param_csv.loc['Adaptive dt','Value'])
        self.true_zeta = float(self.param_csv.loc['True zeta','Value'])
        self.adpt_zeta = float(self.param_csv.loc['Adaptive zeta','Value'])
        self.mu_bar = float(self.param_csv.loc['Mu_bar','Value'])
        self.c_height = float(self.param_csv.loc['Upper Channel Height','Value'])
        self.array_size = float(self.param_csv.loc['Array Time Step Size','Value'])
        if file_type == 'valid':
            self.calc_time = float(self.param_csv.loc['Calculation Time (sec)','Value'])
        elif file_type == 'fail':
            self.calc_time = 0
        self.it_count = float(self.param_csv.loc['Number of Iterations needed for Calculation','Value'])
        self.vert_displ = float(self.param_csv.loc['Vertical Displacement','Value'])
        self.flow_type = str(self.param_csv.loc['Flow Type','Value'])
        self.steric_use = str(self.param_csv.loc['Sterics Use','Value'])
        self.U_centerline = float(self.param_csv.loc['Poiseuille U Centerline','Value'])
        self.kolmogorov_freq = float(self.param_csv.loc['Kolmogorov Frequency','Value'])
        self.brownian_use = str(self.param_csv.loc['Brownian Use','Value'])
        self.param_dict = {"Random Seed": self.random_seed,
                        "Channel Height": self.c_height,
                        "Dt Step size": self.array_size,
                        "Calculation Time": self.calc_time,
                        "Number of Iterations": self.it_count,
                        "True dt": [self.true_dt],
                      "Adaptive dt": self.adpt_dt,
                      "True zeta": self.true_zeta,
                      "Adaptive zeta": self.adpt_zeta,
                      "Mu_bar": self.mu_bar,
                      "Starting Vertical Displacement": self.vert_displ,
                      "Flow Type": self.flow_type,
                      "Sterics Use": self.steric_use,
                      "Brownian Use": self.brownian_use,
                      "Poiseuille U Centerline": self.U_centerline,
                      "Kolmogorov Frequency": self.kolmogorov_freq}
    
        return self.param_dict


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_directory", 
                        help="The main directory where the simulation files reside in",
                    type = str)
    parser.add_argument("output_directory", 
                        help="The directory where the compiled simulation parameters will reside in",
                    type = str)
    
    #Uncommment this line when debugging
    # args = parser.parse_args(['C:/Users/thnguye5/Documents/Simulation_Results/Sterics/00_Simulation_Data/Poiseuille_Flow_Filtered/']) 
    args = parser.parse_args()
    
    
    steric_params = simulation_params(input_directory = args.input_directory,
                      output_directory = args.output_directory)
    steric_params.read_files()
                
    
    