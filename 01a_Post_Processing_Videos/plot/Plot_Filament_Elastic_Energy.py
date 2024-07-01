# -*- coding: utf-8 -*-
"""
FILE NAME:      B__v00_00_Plot_Filament_Elastic_Energy.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):        B__v02_04_Video_Animation.py

DESCRIPTION:    This script will plot the filament length at each time step 
                during the duration of the simulation.

INPUT
FILES(S):       1) .NPY file that contains filament elastic energy at each time step
                for each time step for the duration of the simulation.
                2) .NPY file that contains all time steps used for the simulation.

OUTPUT
FILES(S):       1) .PNG file that shows the filament elastic energy each time step 
                during the duration of the simulation.

INPUT
ARGUMENT(S):    1) Input Directory: The directory that contains the simulation
                length data and time data.
                2) Output Directory: The directory that will contain the output
                file (usually the same as the input directory).
                3) Elastic Peaks: Indices along the elastic energy array
                that indicate maximum filament bending.

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


def plot_fil_elastic_energy(ee_data,time_data,ee_peak_idx,output_directory):
    """
    This function plots the end-to-end length of the filament during the duration of the 
    simulation.
    
    Inputs:
    length_data:            Numpy array that contains the filament length at each time 
                            step during the simulation. 
    time_data:              Numpy that contains all time steps used for the simulation. 
    output_directory:       Directory where the output file will reside in
    """
    create_dir(output_directory)
    
    
    #Regular Elastic Energy
    fig,axes = plt.subplots(figsize = (16,10))
    axes.plot(time_data,ee_data,'gray',linewidth = 1.2)
    if ee_peak_idx:
        for i,v in enumerate(ee_peak_idx):
            axes.plot(time_data[ee_peak_idx[i]],ee_data[ee_peak_idx[i]],color = 'red',marker= 'o',markersize = 10)
        
    
    axes.set_xlabel('Time',fontsize=25,labelpad = 20)
    axes.set_ylabel(r'$E_{elastic}$',fontsize=25,labelpad = 20)
    axes.xaxis.offsetText.set_fontsize(0)
    axes.set_title('Filament Elastic Energy over Simulation',fontsize = 30,pad = 25)
    axes.tick_params(axis='both', which='major', labelsize=12)
    # ax.set_ylim(0,200)
    # ax.set_yticks(np.linspace(0,500,9))
    # axes.ticklabel_format(axis="x", style="sci", scilimits=(-2,-2))
    axes.tick_params(axis='x', which='major', labelsize = 20, size = 6,width = 5)
    axes.tick_params(axis='y', which='major',labelsize = 20, size = 6,width = 5)
    plt.savefig(os.path.join(output_directory,'filament_elastic_energy.png'),dpi = 600,
                bbox_inches = 'tight')
    plt.show()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_directory",
                        help="Specify the parent directory of the simulation data",
                    type = str)
    parser.add_argument("output_directory",
                        help="Specify the directory where the output files will reside in",
                    type = str)
    parser.add_argument("--elastic_energy_peaks","-ee_p",
                        help = "Specify the indices along the elastic energy array that need to be highlighted",
                        nargs = '*',type = int,action = 'store_true',default = False)
    args = parser.parse_args()
    
    ee_array = np.load(os.path.join(args.input_directory,"filament_elastic_energy.npy"))
    time_array = np.load(os.path.join(args.input_directory,"filament_time_vals_all.npy"))
    
    plot_fil_elastic_energy(ee_array,time_array,args.elastic_energy_peaks,args.output_directory)
    

