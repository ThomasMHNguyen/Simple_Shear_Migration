# -*- coding: utf-8 -*-
"""
FILE NAME:      Plot_Depletion_Layer_Thickness.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):        A__v01_00_Process_COM_Migration_Data.py; create_dir.py

DESCRIPTION:    This script will plot the depletion layer thickness as a function
                of mu_bar. 

INPUT
FILES(S):       1) .CSV file that contains the average center of mass position, 
                average true center position, and stress values at each time 
                step during the simulation. 

OUTPUT
FILES(S):       1) .PNG file that shows the depletion layer thickness as a function of mu_bar.
                2) .PDF file that shows the depletion layer thickness as a function of mu_bar.
                3) .EPS file that shows the depletion layer thickness as a function of mu_bar.


INPUT
ARGUMENT(S):    1) Input Ensemble File: The true path to the master .CSV file
                that contains the ensemble average center of mass position, 
                ensemble average true center position, and stress values at 
                each time step during the simulation. 
                2) Main Output directory: The directory that will house all of the
                output files associated with the analysis; if it doesn't exist,
                it will be created.

CREATED:        22Nov22

MODIFICATIONS
LOG:
22Nov22         1) Migrated code to generate the plots from the original script
                to its own instance here.

    
            
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.8.8

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
import pandas as pd
import statsmodels.formula.api as smf
        

def plot_depl_layer_thickness(ensemble_avg_data_df,intercept,slope,data_type,file_name,output_directory):
    """
    This function plots the net displacement of a filament as a function of mu_bar
    based on a particular rigidity profile, channel height, 
    and steric velocity exponential coefficient.
    
    Inputs:
    
    ensemble_avg_data_df:           DataFrame that contains averaged stress tensor data.
    output_directory:           Directory where the generated plots will be stored in.
    """
    ## Calculate Depletion Layer thickness: wall height minus final COM position ##
    fil_df= ensemble_avg_data_df[(ensemble_avg_data_df['Brownian Time'] == 5e-2) &\
                                                                  (ensemble_avg_data_df['Channel Height'] == 0.5)]

    
    
    
    x = np.logspace(3,6,1000)
    y = 10**(intercept)*x**(slope)
    
    fig,axes = plt.subplots(figsize = (7,7))
    sns.lineplot(x = 'Mu_bar',y = 'Distance From Wall',
                  data = fil_df,
                  err_style="bars",
                  marker = 'o',markersize = 5.5,
                  linestyle = '',legend = False,
                  errorbar=("sd", 1),color = 'black')
    plt.plot(x,y,'r',linewidth = 1.2)
    axes = plt.gca()
    axes.set(xscale="log", yscale="log")
    # axes.set_ylim(1.2e-1,4.3e-1)
    axes.set_xlabel(r"$\bar{\mu}$",fontsize = 13,labelpad = 5)
    if data_type == 'average_com':
        # axes.set_ylabel(r"$H - \langle y^{\text{com}}_{f}\rangle$",fontsize = 13,labelpad = 5)
        axes.set_ylabel(r"$L_{d}$",fontsize = 13,labelpad = 5)
    elif data_type == 'ensemble_com':
        axes.set_ylabel(r"$H - y^{\text{com}}_{f}$",fontsize = 13,labelpad = 5)
    axes.tick_params(axis='both', which='both', direction = 'in',labelsize=15,pad = 5)
    axes.set_aspect((np.diff(np.log(axes.get_xlim())))/(np.diff(np.log(axes.get_ylim()))))
    filename_png = '{}.png'.format(file_name)
    filename_pdf = '{}.pdf'.format(file_name)
    filename_eps = '{}.eps'.format(file_name)
    # plt.savefig(os.path.join(output_directory,filename_png),bbox_inches = 'tight',
    #             format = 'png',dpi = 400)
    # plt.savefig(os.path.join(output_directory,filename_pdf),bbox_inches = 'tight',
    #             format = 'pdf',dpi = 400)
    # plt.savefig(os.path.join(output_directory,filename_eps),bbox_inches = 'tight',
    #             dpi = 400,format = 'eps')
    plt.show()
    
    
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_directory",
                        help="Specify the location of the directory where the dependency scripts are located in",
                    type = str)
    parser.add_argument("input_ensemble_file",
                        help="Specify the Absolute Path to the CSV file that contains the ensemble average data",
                    type = str)
    parser.add_argument("output_directory",
                        help = "Specify the directory where the resulting plots will be saved in",
                        type = str)
    args = parser.parse_args()
    os.chdir(args.project_directory)
    from misc.create_dir import create_dir
    
    
    ensemble_data_df = pd.read_csv(args.input_ensemble_file,index_col = 0,header = 0)
    create_dir(args.output_directory)
    
    plot_depl_layer_thickness(ensemble_data_df,args.output_directory)

