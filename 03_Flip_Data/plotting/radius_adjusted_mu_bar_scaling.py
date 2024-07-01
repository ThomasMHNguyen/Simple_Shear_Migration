# -*- coding: utf-8 -*-
"""
FILE NAME:              radius_adjusted_mu_bar_scaling.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):                A__v01_03_Process_Filament_Flip_Data.py

DESCRIPTION:            This script contains a function that will plot the radius of 
                        bending of the J-shapes as a function of the adjusted mu_bar value
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

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

### Use LaTeX to generate plots ###
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})


#%%
def radius_adjusted_mu_bar_scaling(input_file,output_directory,file_name):
    """
    This function will take in the filament flipping data and plot the radius 
    of bending as a function of adjusted mu_bar.
    
    Inputs:
    
    input_file:                 Input Pandas dataframe that lists the measured
                                data values for each U-turn flipping event.
    output_directory:           Path to directory to be created.
    file_name:                  The file name to be used for the output graphical
                                files.
    """
    hue_order_fix = [r'$5 \times 10^{4}$',r'$1 \times 10^{5}$',
                 r'$2 \times 10^{5}$',r'$5 \times 10^{5}$']

    color_palette = ["#EFA8B8","#E26D5A","#0F7173","#4A4063"]
    if np.where(input_file['Mu_bar'].unique() == 5e5)[0]:
        hue_order_fix = hue_order_fix
    
        color_palette = color_palette
    else:
        hue_order_fix = hue_order_fix[:-1]
        color_palette = color_palette[:-1]
    ### log scale, emphasis on flow type
    x = np.linspace(1.3e2,1e6,100)
    y1 = 0.73*x**-0.25
    fig,axes = plt.subplots(figsize = (7,7))
    
    sns.scatterplot(x = 'Mu_bar times Starting COM-y',y = 'Radius of Bending-1',
                               data = input_file,hue = 'Mu_bar String',
                               palette = color_palette,
                               hue_order = hue_order_fix,
                               alpha = 0.7)
    # plt.plot(x,y1,color = 'black',linestyle = 'dashed',
    #          linewidth = 1.5,
    #                   label = r'$R \sim \left(\bar{\mu}y_{i}^{\text{com}}\right)^{-1/4}$')
    axes.set_ylim(2.5e-2,2e-1)
    axes.set_xlim(1e3,1.2e6)
    axes.set(xscale = 'log',yscale = 'log')
    axes.tick_params(axis='both', which='both', direction = 'in',labelsize=11,pad = 5)

    axes.set_aspect((np.log(axes.get_xlim()[1]) - np.log(axes.get_xlim()[0]))/(np.log(axes.get_ylim()[1]) - np.log(axes.get_ylim()[0])))
    axes.set_xlabel(r"$\bar{\mu}y_{i}^{\text{com}}$",fontsize = 13,labelpad = 5)
    axes.set_ylabel(r"$R$",fontsize = 13,labelpad = 5)
    axes.legend(loc='lower left', 
                prop={'size': 12},title= r"$\bar{\mu}$").get_title().set_fontsize("13")
    # fig.savefig(os.path.join(output_directory,'{}.png'.format(file_name)),
    #             bbox_inches = 'tight',dpi = 200)
    # fig.savefig(os.path.join(output_directory,'{}.pdf'.format(file_name)),
    #             bbox_inches = 'tight',dpi = 200)
    # fig.savefig(os.path.join(output_directory,'{}.eps'.format(file_name)),
    #             bbox_inches = 'tight',format = 'eps',dpi = 200)
    plt.show()