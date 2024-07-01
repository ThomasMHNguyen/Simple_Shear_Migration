# -*- coding: utf-8 -*-
"""
FILE NAME:      B__v00_00_Plot_Filament_Stresses.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):        B__v02_04_Video_Animation.py; create_dir.py

DESCRIPTION:    This script will plot the filament length at each time step 
                during the duration of the simulation.

INPUT
FILES(S):       1) .NPY file that contains filament length at each time step
                for each time step for the duration of the simulation.
                2) .NPY file that contains all time steps used for the simulation.

OUTPUT
FILES(S):       1) .PNG file that shows the filament length each time step 
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


def plot_fil_stresses(stress_data,time_data,output_directory):
    """
    This function plots the end-to-end length of the filament during the duration of the 
    simulation.
    
    Inputs:
    stress_data:            Numpy array that contains the filament length at each time 
                            step during that contains averaged extra particle stress tensor
                            component values for each time step for 
                            the duration of the simulation.
    time_data:              Numpy that contains all time steps used for the simulation. 
    output_directory:       Directory where the output file will reside in
    """
    create_dir(output_directory)
    
    
    fig,axes = plt.subplots(nrows = 3,figsize = (24,18))
    # fig.tight_layout()
    axes[0].plot(time_data,
                  stress_data[0,0,:] - stress_data[1,1,:],'red',
                  linewidth = 2,
                  label = r"$\mathbf{\Sigma}_{11} - \mathbf{\Sigma}_{22}$") #First normal stress
    axes[1].plot(time_data,
                  stress_data[1,1,:] - stress_data[2,2,:],'red',
                  linewidth = 2,
                  label = r"$\mathbf{\Sigma}_{22} - \mathbf{\Sigma}_{33}$") #second normal stress
    axes[2].plot(time_data,
                  stress_data[0,1,:],'red',
                  linewidth = 2,
                  label = r"$\mathbf{\Sigma}_{12}$") #shear stress
    axes[2].plot(time_data,
                  stress_data[1,0,:],'blue',
                  linewidth = 2,
                  label = r"$\mathbf{\Sigma}_{21}$") #shear stress
    
    # axes[0].set_ylim(-1500,1500)
    # axes[1].set_ylim(-1500,1500)
    # axes[2].set_ylim(-1500,1500)
    
    # axes[0].set_xlim(-1e-4,5.01e-2)
    # axes[1].set_xlim(-1e-4,5.01e-2)
    # axes[2].set_xlim(-1e-4,5.01e-2)
    
    # axes[0].set_xlim(35,65)
    # axes[1].set_xlim(35,65)
    # axes[1].set_ylim(-200,400)
    # axes[2].set_xlim(35,65)
    # axes[2].set_ylim(-30,400)
    fig.suptitle("Filament Stresses",fontsize = 30,y = 0.95)
    axes[0].xaxis.offsetText.set_fontsize(18)
    axes[1].xaxis.offsetText.set_fontsize(18)
    axes[2].xaxis.offsetText.set_fontsize(18)
    
    # axes[0].axvline(x = 2.732,color = 'gray',
    #             linestyle = 'dashdot',linewidth = 3)
    # axes[1].axvline(x = 2.732,color = 'gray',
    #             linestyle = 'dashdot',linewidth = 3)
    # axes[2].axvline(x = 2.732,color = 'gray',
    #             linestyle = 'dashdot',linewidth = 3)
    # axes[0].set_aspect((7)/(2.5*3000))
    # axes[1].set_aspect((7)/(2.5*3000))
    # axes[2].set_aspect((7)/(2.5*3000))
    
    axes[0].legend(loc='center left',bbox_to_anchor=(1, 0.5),prop={'size': 17})  
    axes[1].legend(loc='center left',bbox_to_anchor=(1, 0.5),prop={'size': 17})  
    axes[2].legend(loc='center left',bbox_to_anchor=(1, 0.5),prop={'size': 17}) 
    # axes[0].set_title(r"Normal Stress Difference $\mathbf{N}_{1}$",fontsize = 15,pad = 5)  
    # axes[1].set_title(r"Normal Stress Difference $\mathbf{N}_{2}$",fontsize = 15,pad = 5)
    # axes[2].set_title(r"Shear Stress$",fontsize = 15,pad = 5)
    axes[0].set_ylabel("Stress Difference",fontsize = 25,labelpad = 15)
    # axes[0].set_xlabel("Time",fontsize = 15,labelpad = 5)
    axes[1].set_ylabel("Stress Difference",fontsize = 25,labelpad = 15)
    # axes[1].set_xlabel("Time",fontsize = 15,labelpad = 5)
    axes[2].set_ylabel("Stress",fontsize = 25,labelpad = 25)
    axes[2].set_xlabel("t^{Br}",fontsize = 25,labelpad = 10)
    
    axes[0].tick_params(axis='x', which='major', labelsize = 20, size = 6,width = 5)
    axes[0].tick_params(axis='y', which='major',labelsize = 20, size = 6,width = 5)
    axes[1].tick_params(axis='x', which='major', labelsize = 20, size = 6,width = 5)
    axes[1].tick_params(axis='y', which='major',labelsize = 20, size = 6,width = 5)
    axes[2].tick_params(axis='x', which='major', labelsize = 20, size = 6,width = 5)
    axes[2].tick_params(axis='y', which='major',labelsize = 20, size = 6,width = 5)
    
    
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    axes[2].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    plt.savefig(os.path.join(output_directory,'filament_stress_differences.png'),dpi = 600,
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
    args = parser.parse_args()
    
    stress_array = np.load(os.path.join(args.input_directory,"filament_stress_all.npy"))
    time_array = np.load(os.path.join(args.input_directory,"filament_time_vals_all.npy"))
    
    plot_fil_stresses(stress_array,time_array,args.output_directory)
    
    