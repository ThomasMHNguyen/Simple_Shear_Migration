# -*- coding: utf-8 -*-
"""
FILE NAME:      Parse_Simulation_Data.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    This script will filter and decrease the resolution of data in 
                the simulations; the filtered data will be written to a new directory.
                

INPUT
FILES(S):       1) .NPY files that contain numerical data corresponding to a simulation
                run. 
                2) .CSV file that stores all of the parameters used for a simulation.

OUTPUT
FILES(S):       1) .NPY files that contain numerical data corresponding to a simulation
                run. 
                2) .CSV file that stores all of the parameters used for a simulation.


INPUT
ARGUMENT(S):    1) Main Input directory: The directory that houses all of the simulation
                data files.
                2) Output directory: The directory that will house the filtered
                simulation data files.


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
import re, os, argparse, shutil
import numpy as np
import pandas as pd



def parse_data(save_time,end_time,input_directory,output_directory):
    """
    This function filters the simulation data based on a specified time step size
    and writes the filtered data to a new directory.
    
    Inputs:
        
    save_time:              The time step size you want to filter the data for.
    end_time:               The final time of the simulation data.
    input_directory:        The directory that contains the unfiltered 
                            simulation data.
    output_directory:       The directory where the filtered simulation data 
                            will be saved to.
    """
    #Read the long time data & specify time step size to filter for
    time_vals_all = np.load(os.path.join(input_directory,'filament_time_vals_all.npy'))
    desired_time_vals = np.round(np.linspace(0,end_time,int(end_time/save_time)+1),
                                 int(-np.log10(save_time)))
    
    ### Find indices where old time value index is in ###
    new_index_vals_save = np.isin(time_vals_all,desired_time_vals).nonzero()[0]
    
    ### Load other arrays ###
    filament_position_data = np.load(os.path.join(input_directory,'filament_allstate.npy'))
    filament_stress_data = np.load(os.path.join(input_directory,'filament_stress_all.npy'))
    filament_tension_data = np.load(os.path.join(input_directory,'filament_tension.npy'))
    filament_elastic_data = np.load(os.path.join(input_directory,'filament_elastic_energy.npy'))
    filament_length_data = np.load(os.path.join(input_directory,'filament_length.npy'))
    filament_angle_data = np.load(os.path.join(input_directory,'filament_angles.npy'))
    filament_deflection_data = np.load(os.path.join(input_directory,'filament_deflection.npy'))
    
    filament_position_data = filament_position_data[:,:,new_index_vals_save]
    filament_stress_data = filament_stress_data[:,:,new_index_vals_save]
    filament_tension_data = filament_tension_data[:,new_index_vals_save]
    filament_elastic_data = filament_elastic_data[new_index_vals_save]
    filament_length_data = filament_length_data[new_index_vals_save]
    filament_angle_data = filament_angle_data[new_index_vals_save]
    filament_deflection_data = filament_deflection_data[new_index_vals_save]
    
    #Save filtered data
    # np.save(os.path.join(output_directory,'filament_allstate.npy'),filament_position_data)
    # np.save(os.path.join(output_directory,'filament_stress_all.npy'),filament_stress_data)
    # np.save(os.path.join(output_directory,'filament_tension.npy'),filament_tension_data)
    # np.save(os.path.join(output_directory,'filament_elastic_energy.npy'),filament_elastic_data)
    # np.save(os.path.join(output_directory,'filament_length.npy'),filament_length_data)
    # np.save(os.path.join(output_directory,'filament_angles.npy'),filament_angle_data)
    # np.save(os.path.join(output_directory,'filament_deflection.npy'),filament_deflection_data)
    # np.save(os.path.join(output_directory,'filament_time_vals_all.npy'),desired_time_vals)
    
    ### Modify Parameter Values ###
    param_csv = pd.read_csv(os.path.join(input_directory,'parameter_values.csv'))
    param_csv['Array Time Step Size','Value'] = save_time
    param_csv['Simulation End Time','Value'] = end_time
    # param_csv.to_csv(os.path.join(output_directory,'parameter_values.csv'))
    
def delete_dir(directory):
    """
    This function deletes the specified directory and all of the contents in it.
    """
    shutil.rmtree(directory)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_directory", 
                        help="The main directory where the simulation files currently reside in",
                    type = str)
    parser.add_argument("output_directory", 
                        help="The new directory where the simulation files will reside in",
                    type = str)
    #uncomment this line for debugging
    args = parser.parse_args(['C:/Users/thnguye5/Documents/Simulation_Results/01_Sterics/00a_Simulation_Data/Shear_Flow/',
                               'C:/Users/thnguye5/Documents/Simulation_Results/01_Sterics/00a_Simulation_Data/Shear_Flow_Filtered/'])
    # args = parser.parse_args()
    
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    old_new_path_comp_list = []
    dirs_to_remove = []
    for root, dirs, files in os.walk(args.input_directory):
        for dir_ in dirs:
            cur_path = os.path.join(root,dir_)
            check_file = os.path.join(cur_path,'filament_allstate.npy')
            if os.path.exists(check_file):
                diff_path = os.path.relpath(cur_path,args.output_directory)
                new_path = os.path.join(args.output_directory,
                                        *(diff_path.split(os.path.sep)[2:]))
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                    
                param_csv = pd.read_csv(os.path.join(cur_path,'parameter_values.csv'),
                                        index_col = 0,header = 0)
                curr_timestep = float(param_csv.loc['Array Time Step Size','Value']) #Use time step size for saved data
                final_time = float(param_csv.loc['Simulation End Time','Value']) #Use actual length of simulation end time
                # final_time = 5e-2 #Specify actual end time 
                # parse_data(save_time = 1e-5,
                #            end_time = final_time,
                #            input_directory = cur_path,
                #            output_directory = new_path)
                
                #Delete the data if the time step size is too large
                if curr_timestep != 1e-6:
                    dirs_to_remove.append(cur_path)
                    delete_dir(cur_path)
                
                ### Check to make sure final directory is appropriately named ###
                # path_change_dict = {"Old Directory":[cur_path],
                #                     "New Directory": [new_path]}
                # old_new_path_comp_list.append(path_change_dict)
                
    ### Save New Directory name changes to a CSV File ###
    # old_new_path_df = pd.concat(
    #     [pd.DataFrame.from_dict(i) for i in old_new_path_comp_list],
    #     ignore_index = True)
    # old_new_path_df.to_csv(os.path.join(args.output_directory,'directory_changes.csv'))
                