# -*- coding: utf-8 -*-
"""
FILE NAME:      B__v00_00_Plot_Filament_EE_Length.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):        B__v02_04_Video_Animation.py

DESCRIPTION:    This script will plot the filament end-to-end length at each time step 
                during the duration of the simulation.

INPUT
FILES(S):       1) .NPY file that contains filament position at each time step
                for each time step for the duration of the simulation.
                2) .NPY file that contains all time steps used for the simulation.

OUTPUT
FILES(S):       1) .PNG file that shows the filament end-to-end length each time step 
                during the duration of the simulation.

INPUT
ARGUMENT(S):    1) Input Directory: The directory that contains the simulation
                length data and time data.
                2) Output Directory: The directory that will contain the output
                file (usually the same as the input directory).

CREATED:        22Jan23

MODIFICATIONS
LOG:            N/A
            
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.9.13

VERSION:        1.0

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:     N/A

NOTE(S):        N/A

"""
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Import other files & plotting features ###
from create_dir import create_dir


def plot_fil_ee_length(position_data,time_data,output_directory):
    """
    This function plots the end-to-end length of the filament during the duration of the 
    simulation.
    
    Inputs:
    position_data:          Numpy array that contains the filament position at each time 
                            step during the simulation. 
    time_data:              Numpy that contains all time steps used for the simulation. 
    output_directory:       Directory where the output file will reside in
    """
    
    create_dir(output_directory)
    
    ee_length = np.sqrt(np.sum((position_data[-1,:,:] - position_data[0,:,:])**2,axis = 0))
    
    fig,axes = plt.subplots(figsize = (7,7))
    plt.plot(time_data,ee_length,color = 'black',linewidth = 1.2)
    axes.ticklabel_format(axis="x", style="sci", scilimits=(-2,-2))
    axes.xaxis.offsetText.set_fontsize(0)
    axes.set_xlabel(r'$t^{Br}$',fontsize=25,labelpad = 20)
    axes.set_ylabel(r'$L$',fontsize=25,labelpad = 20)
    axes.set_title(r'$L$ vs. $t^{Br}$',fontsize = 30,pad = 25)
    axes.set_ylim(0.60,1.02)
    axes.tick_params(axis='x', which='major', labelsize = 14, size = 7,width = 1.5)
    axes.tick_params(axis='y', which='major',labelsize = 14, size = 7,width = 1.5)
    plt.savefig(os.path.join(output_directory,'filament_length_change.png'),
                dpi = 400,bbox_inches = 'tight')
    plt.show()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_directory",
                        help="Specify the parent directory of the simulation data",
                    type = str)
    parser.add_argument("output_directory",
                        help="Specify the directory where the output files will reside in",
                    type = str)
    args = parser.parse_args()
    
    position_array = np.load(os.path.join(args.input_directory,"filament_allstate.npy"))
    time_array = np.load(os.path.join(args.input_directory,"filament_time_vals_all.npy"))
    
    plot_fil_ee_length(position_array,time_array,args.output_directory)

