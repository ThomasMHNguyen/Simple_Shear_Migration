# -*- coding: utf-8 -*-
"""
FILE NAME:              radius_mu_bar_scaling.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):                A__v01_03_Process_Filament_Flip_Data.py

DESCRIPTION:            This script contains a function that will plot the radius of 
                        bending of the J-shapes as a function of the mu_bar value
                        (mu_bar times center of mass prior to rotation).

INPUT
FILES(S):               N/A

OUTPUT
FILES(S):               1) .PNG/.PDF/.EPS file that shows the scatterplot relationship between
                        radius of bending and adjusted mu_bar.
                

INPUT
ARGUMENT(S):            N/A

CREATED:                22Nov22

MODIFICATIONS
LOG:
22Nov22                 1) Migrated code to generate the plots from the original script
                        to its own instance here.

    
            
LAST MODIFIED
BY:                     Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:                 3.8.8

VERSION:                1.0

AUTHOR(S):              Thomas Nguyen

STATUS:                 Working

TO DO LIST:             N/A

NOTE(S):                N/A
"""

import  os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

### Use LaTeX to generate plots ###
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})
#%%

def radius_mu_bar_scaling(input_file,output_directory,flow_type,file_name):
    """
    This function will take in the filament flipping data and plot the radius 
    of bending as a function of mu_bar.
    
    Inputs:
    
    input_file:                 Input Pandas dataframe that lists the measured
                                data values for each U-turn flipping event.
    output_directory:           Path to directory to be created.
    flow_type:                  String identifier to denote what kind of background flow.
    file_name:                  The file name to be used for the output graphical
                                files.
    """

    input_file['Data Type'] = 'Measured'
    
    ### log scale, emphasis on flow type
    if flow_type == 'Shear':
        x = np.linspace(1e4,5e5,100)
        y1 = 1.32*x**-0.25
    else:
        x = np.linspace(1e4,1e6,100)
        y1 = 1.10*x**-0.25
    fig,axes = plt.subplots(figsize = (6,6))
    
    
    sns.lineplot(x = 'Mu_bar',y = 'Radius of Bending-1',
                               data = input_file,
                               palette = ['#E54B4B'],
                               marker = 'o',markersize = 5.5,
                               err_style="bars",hue = 'Data Type',
                               linestyle = '',
                               errorbar=("sd", 1))
    plt.plot(x,y1,color = 'black',linewidth = 0.9,linestyle = 'dashed',
             label = r'$R \sim \bar{\mu}^{-1/4}$')
    axes.set(xscale = 'log',yscale = 'log')
    axes.tick_params(axis='both', which='both',direction = 'in', labelsize=12)
    # axes.xaxis.offsetText.set_fontsize(0)

    
    axes.set_xlabel(r"$\bar{\mu}$",fontsize = 13,labelpad = 5)
    axes.set_ylabel(r"$R$",fontsize = 13,labelpad = 5)
    axes.legend(loc='upper right', 
                prop={'size': 12},title= r"$\text{Data Type}$").get_title().set_fontsize("13")
    axes.set_aspect((np.diff(np.log(axes.get_xlim())))/(np.diff(np.log(axes.get_ylim()))))
    # fig.savefig(os.path.join(output_directory,'{}.png'.format(file_name)),
    #             bbox_inches = 'tight',dpi = 200)
    # fig.savefig(os.path.join(output_directory,'{}.pdf'.format(file_name)),
    #             bbox_inches = 'tight',dpi = 200)
    # fig.savefig(os.path.join(output_directory,'{}.eps'.format(file_name)),
    #             bbox_inches = 'tight',format = 'eps',dpi = 200)
    plt.show()
