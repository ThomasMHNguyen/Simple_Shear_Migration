# -*- coding: utf-8 -*-
"""
FILE NAME:              depletion_layer.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):                Process_COM_Migration_Data.py


DESCRIPTION:            This script contains a function that will plot the 
                        ensemble average depletion as a function of mu_bar.
                    

INPUT
FILES(S):               N/A

OUTPUT
FILES(S):               1) .PNG/.PDF/.EPS file that shows the depletion layer 
                        scaling.

INPUT
ARGUMENT(S):            N/A

CREATED:                22Jan23

MODIFICATIONS
LOG:                    N/A
                
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

import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})
# plt.rcdefaults()

def depletion_layer(ensemble_avg_data_df,params_dict,data_type,
                              file_name,output_directory):
    """
    This function plots the net displacement of a filament as a function of mu_bar
    based on a particular rigidity profile, channel height, 
    and steric velocity exponential coefficient.
    
    Inputs:
    
    ensemble_avg_data_df:       DataFrame that contains the calculated depletion
                                layer distance based on the average ensemble data.
    params_dict:                Dictionary that contains the fitted data parameters.
    data_type:                  String that specifies whether to plot the ensemble
                                average data or the ensemble data.
    file_name:                  Name of the resulting saved files.
    output_directory:           Directory where the generated plots will be stored in.
    """
    ## Calculate Depletion Layer thickness: wall height minus final COM position ##
    fil_df= ensemble_avg_data_df[
        (ensemble_avg_data_df['Brownian Time'] == 5e-2) &\
            (ensemble_avg_data_df['Channel Height'] == 0.25)]
        
    slope,intercept = params_dict['Slope'],params_dict['Intercept']
    x = np.logspace(3,6,1000)
    y = (intercept)*x**(slope)
    
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
    plt.savefig(os.path.join(output_directory,filename_png),bbox_inches = 'tight',
                format = 'png',dpi = 400)
    plt.savefig(os.path.join(output_directory,filename_pdf),bbox_inches = 'tight',
                format = 'pdf',dpi = 400)
    plt.savefig(os.path.join(output_directory,filename_eps),bbox_inches = 'tight',
                dpi = 400,format = 'eps')
    plt.show()