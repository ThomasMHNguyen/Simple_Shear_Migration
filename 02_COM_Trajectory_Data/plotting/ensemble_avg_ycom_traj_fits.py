# -*- coding: utf-8 -*-
"""
FILE NAME:              ensemble_avg_ycom_traj_fits.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):                Process_COM_Migration_Data.py

DESCRIPTION:            This script will plot the fitted model to the filament ensemble
                        average center of mass position vs. time. 

INPUT
FILES(S):               N/A

OUTPUT
FILES(S):               .PNG/.PDF/.EPS file that shows the ensemble average trajectory with 
                        the fitted exponential model data as subplots


INPUT
ARGUMENT(S):            N/A

CREATED:                24Apr23

MODIFICATIONS
LOG:                    N/A

                
LAST MODIFIED
BY:                     Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:                 3.9.13

VERSION:                1.0

AUTHOR(S):              Thomas Nguyen

STATUS:                 Working

TO DO LIST:             N/A

NOTE(S):                N/A

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": "Times",
#     'text.latex.preamble': r'\usepackage{amsmath}'})

plt.rcdefaults()



def ensemble_avg_ycom_traj_fits(ensemble_data_df,output_dir):
    """
    This function plots the ensemble average center of mass trajectory of a 
    filament based on a particular rigidity profile, 
    flow strength value, channel height, starting vertical displacement, 
    steric velocity exponential coefficient, flow type, and whether or not
    the sterics algorithm was used or not. On the adjacent plot, it will plot 
    
    Inputs:
        
    ensemble_data_df:           DataFrame that contains all ensemble
                                average data.
    output_directory:           Directory where the generated plots will be stored in.
    """

    fil_data_df = ensemble_data_df[ensemble_data_df['Height Class'] != 'Middle']
    #Group by Rigidity Profile, Mu_bar,Channel Height, Vertical Displacement, Velocity Exponential Coefficient
    exp_groups = fil_data_df.groupby(
        by = ['Rigidity Suffix','Channel Height',  #0-1
              'Poiseuille U Centerline','Kolmogorov Phase Text','Kolmogorov Phase Value','Kolmogorov Frequency', #2-5
              'Steric Velocity Exponential Coefficient','Steric Velocity Gap Criteria', #6-7
              'Sterics Use','Flow Type']) #8-9
    for group in exp_groups.groups.keys():
        group_df = exp_groups.get_group(group)
        rigid,channel_h,u_centerline,k_phase_text,k_phase_val,k_freq = group[:6]
        steric_use,flow_type,height_class = group[-3:]
        
        group_df1 = group_df[group_df['Fit Parameter'] != 'T_0']
        group_df2 = group_df[group_df['Fit Parameter'] == 'T_0']
        group_df2['ABS T_0'] = np.abs(group_df2['T_0'])
        
        ##### Center of Mass #####
        fig,axes = plt.subplots(nrows = 2,ncols = 2,figsize = (7,7),sharey = True)
        sns.scatterplot(y = 'Value',x = 'Mu_bar',hue = 'Fit Parameter',
                                palette = 'bright',data = group_df1[group_df1['Height Class'] == 'Low'],
                                style = 'Starting Vertical Displacement',
                                alpha = 0.5,ax = axes[0,0])
        sns.scatterplot(y = 'Value',x = 'Mu_bar',hue = 'Fit Parameter',
                                palette = 'bright',data = group_df1[group_df1['Height Class'] == 'High'],
                                style = 'Starting Vertical Displacement',
                                alpha = 0.5,ax = axes[0,1])
        sns.scatterplot(y = 'Value',x = 'Mu_bar',hue = 'Fit Parameter',
                                palette = 'bright',data = group_df2[group_df2['Height Class'] == 'Low'],
                                style = 'Starting Vertical Displacement',
                                alpha = 0.5,ax = axes[1,0])
        sns.scatterplot(y = 'Value',x = 'Mu_bar',hue = 'Fit Parameter',
                                palette = 'bright',data = group_df2[group_df2['Height Class'] == 'High'],
                                style = 'Starting Vertical Displacement',
                                alpha = 0.5,ax = axes[1,1])
        
        for n_row,ax_row in axes:
            for n_col,ax_col in ax_row:
                if n_row == 0:
                    ax_col.tick_params(axis='both', which='major',direction = 'in', labelsize=11)
                    ax_col.set_xlabel("")
                elif n_row == 1:
                    ax_col.tick_params(axis='both', which='major',direction = 'in', labelsize=11)
                    # ax.ticklabel_format(axis="x", style="sci", scilimits=(4,4))
                    # ax.xaxis.offsetText.set_fontsize(0)
                    ax_col.set(xscale = 'log',yscale = 'log')
                    ax_col.set_xlabel("")
                    # axes.set_ylim(-1.1,1.1)
                    # axes.set_yticks(np.linspace(-1,1,5))
                    # axes.set_xlim(3e4,5.2e5)
                    # axes.set_xticks(np.linspace(1e5,5e5,5))
                    ax_col.set_aspect(np.diff(np.log10(ax_col.get_xlim()))/(np.diff(np.log10(ax_col.get_ylim()))))
                    ax_col.legend(loc = 'upper right')
        fig.supxlabel(r"$\bar{\mu}$",size = 13,y = 0.20)
        
            
        ### Save figure ###
        filename_png = '{}.png'.format(filename_prefix)
        filename_pdf = '{}.pdf'.format(filename_prefix)
        filename_eps = '{}.eps'.format(filename_prefix)
        
        plt.savefig(os.path.join(output_dir,filename_png),bbox_inches = 'tight',
                    dpi = 200)
        plt.savefig(os.path.join(output_dir,filename_pdf),bbox_inches = 'tight',
                    dpi = 200)
        plt.savefig(os.path.join(output_dir,filename_eps),bbox_inches = 'tight',
                    dpi = 200,format = 'eps')
        plt.show()
