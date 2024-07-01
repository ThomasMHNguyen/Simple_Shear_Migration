# -*- coding: utf-8 -*-
"""
FILE NAME:              ycom_distr_snps_poi.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):                Process_COM_Migration_Data.py


DESCRIPTION:            This script will plot the probability distribution of 
                        all filament ensembles' y-COM at various time points in 
                        Poiseuille flow at 2 mu_bar values. 


INPUT
FILES(S):               N/A

OUTPUT
FILES(S):               1) .PNG/.PDF/.EPS file that shows the probability distribution 
                        of the y-COM at various timepoints.



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

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})


def ycom_distr_snps_poi(ensemble_data_df,mu_bar_1,mu_bar_2,time_vals,
                        time_vals_text,output_dir):
    
    """
    This function creates a series of subplots that show: 1) the velocity and
    velocity gradient profile of Kolmogorov flow; 2) the y-center of mass distribution
    of all ensemble simulations at various points in time.
    
    Inputs:
        
    ensemble_data_df:               Pandas DataFrame that contains all ensemble
                                    data for Kolmogorov flow.
    mu_bar_1:                       One of the Mu_bar values to plot the data for.
    mu_bar_2:                       One of the Mu_bar values to plot the data for.
    time_vals:                      Numpy array of 5 time values to plot the 
                                    y-center of mass distribution data for.
    time_vals_text:                 List of 5 time values in text form that will
                                    be shown on the plot.
    output_dir:                     Directory where the plots will be saved in.
    """
    channel_height,u_centerline = 0.5,1
    U_c_text = '{:.2f}'.format(u_centerline).replace('.','p')
    channel_h_text = '{:.2f}'.format(channel_height).replace('.','p')
    
    filename_prefix = '{}_ST_{}_H_{}_UC_{}_sbp'.format("Yes",
                                                         "POI",
                                                         channel_h_text,
                                                         U_c_text)
    fil1_df = ensemble_data_df[(ensemble_data_df['Mu_bar'] == mu_bar_1) & (ensemble_data_df['Poiseuille U Centerline'] == u_centerline) &\
                               (ensemble_data_df['Channel Height'] == channel_height)]
    fil2_df = ensemble_data_df[(ensemble_data_df['Mu_bar'] == mu_bar_2) & (ensemble_data_df['Poiseuille U Centerline'] == u_centerline) &\
                               (ensemble_data_df['Channel Height'] == channel_height)]
        
    time_val_0_data_1 = fil1_df[fil1_df['Brownian Time'] == time_vals[0]]
    time_val_1_data_1 = fil1_df[fil1_df['Brownian Time'] == time_vals[1]]
    time_val_2_data_1 = fil1_df[fil1_df['Brownian Time'] == time_vals[2]]
    time_val_3_data_1 = fil1_df[fil1_df['Brownian Time'] == time_vals[3]]
    time_val_4_data_1 = fil1_df[fil1_df['Brownian Time'] == time_vals[4]]

    time_val_all_data_1 = [time_val_0_data_1,time_val_1_data_1,time_val_2_data_1,
                           time_val_3_data_1,time_val_4_data_1]

    time_val_0_data_2 = fil2_df[fil2_df['Brownian Time'] == time_vals[0]]
    time_val_1_data_2 = fil2_df[fil2_df['Brownian Time'] == time_vals[1]]
    time_val_2_data_2 = fil2_df[fil2_df['Brownian Time'] == time_vals[2]]
    time_val_3_data_2 = fil2_df[fil2_df['Brownian Time'] == time_vals[3]]
    time_val_4_data_2 = fil2_df[fil2_df['Brownian Time'] == time_vals[4]]

    time_val_all_data_2 = [time_val_0_data_2,time_val_1_data_2,time_val_2_data_2,
                           time_val_3_data_2,time_val_4_data_2]
        
    fig,axes = plt.subplots(nrows = 2,ncols = 5,figsize=(10,7),sharey = True)
    plt.subplots_adjust(hspace=0.4,wspace = -0.845)
    
          
    ### Plot COM Distribution in the subplots grid ###
    for n_row,ax_row in enumerate(axes):
        for n_col,ax_col in enumerate(ax_row):
            if n_row == 0:
                use_data = time_val_all_data_1
            elif n_row == 1:
                use_data = time_val_all_data_2
            if n_col == 0:
                sns.histplot(data = use_data[n_col],
                                y = 'Center of Mass-y',stat = 'density',
                                fill = 'True',alpha = 0.2,linewidth = 0,
                                # binwidth = (time_val_0_data['Center of Mass-y'].max() - time_val_0_data['Center of Mass-y'].min())/\
                                    # time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                                bins = 9,
                                kde = True,
                                ax = ax_col,legend = False,color = "#7209B7")
            else:
                sns.histplot(data = use_data[n_col],
                                y = 'Center of Mass-y',stat = 'density',
                                fill = 'True',alpha = 0.2,linewidth = 0,
                                binwidth = (use_data[n_col]['Center of Mass-y'].max() - use_data[n_col]['Center of Mass-y'].min())/\
                                    use_data[n_col]['Starting Vertical Displacement'].unique().shape[0],
                                # bins = 9,
                                kde = True,
                                ax = ax_col,legend = False,color = "#7209B7")
    
    #Format axes
    for n_row,ax_row in enumerate(axes):
        for n_col,ax_col in enumerate(ax_row):
            # ax_col.set_xlabel(r"Density",fontsize = 13,labelpad = 1)
            ax_col.set_ylim(-0.05,np.round(1.1*channel_height,2))
            ax_col.set_xlim(-0.25,14)
            ax_col.set_yticks(np.linspace(0,0.5,6))
            ax_col.set_xticks(np.linspace(0,10,5))
            ax_col.xaxis.set_major_formatter("{x:.0f}")
            ax_col.tick_params(axis='both', which='major',direction = 'in', labelsize=11,pad = 5)
            [l.set_visible(False) for (i,l) in enumerate(ax_col.xaxis.get_ticklabels()) if (i) % 2 != 0]
            if n_col == 0:
            # Format each subplot in the subplot grid #
                ax_col.set_ylabel(r"$y^{\text{com}}$",fontsize = 15,labelpad = 5)
            if n_col == 2:
               ax_col.set_xlabel("Density",fontsize = 15,labelpad = 5)
            else:
                ax_col.set_xlabel("")
            if n_col != 4:
                ax_col.spines['right'].set_visible(False)
            if n_col == 1 or n_col == 3:
                ax_col.set_title(r"${0:.3f}$".format(time_vals[n_col]),
                                    fontsize = 12,pad = 4)
            else:
                ax_col.set_title(r"${0:.2f}$".format(time_vals[n_col]),
                                    fontsize = 12,pad = 4)
            ax_col.set_aspect(70*np.diff(ax_col.get_xlim())/(np.diff(ax_col.get_xlim())))
    
    ### Label x-axes label for all plots
    
    ### text to denote subfigure ###
    axes[0,0].text(x = -14,y = 0.55,s = r'\textbf{(a)}',size = 15)
    axes[1,0].text(x = -14,y = 0.55,s = r'\textbf{(b)}',size = 15)
    ### Save figure-COM ###
        
    filename_png = '{}.png'.format(filename_prefix)
    filename_pdf = '{}.pdf'.format(filename_prefix)
    filename_eps = '{}.eps'.format(filename_prefix)
    
    fig.savefig(os.path.join(output_dir,filename_png),bbox_inches = 'tight',
                dpi = 400)
    fig.savefig(os.path.join(output_dir,filename_pdf),bbox_inches = 'tight',
                format = 'pdf',dpi = 400)
    fig.savefig(os.path.join(output_dir,filename_eps),bbox_inches = 'tight',
                dpi = 400,format = 'eps')
    plt.show()