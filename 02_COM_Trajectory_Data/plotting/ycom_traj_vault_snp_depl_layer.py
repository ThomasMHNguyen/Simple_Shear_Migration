# -*- coding: utf-8 -*-
"""
FILE NAME:              ycom_traj_vault_snp_depl_layer.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):                Process_COM_Migration_Data.py


DESCRIPTION:            This script contains a function plots the following 3 
                        things: 1) Instantaneous average y-center of mass as a 
                        function of time for 2 mu_bar values; 2) Snapshots that
                        show the filament behavior near the wall for these 2 
                        mu_bar values; and 3) scaling relationship between the 
                        depletion layer and mu_bar.


INPUT
FILES(S):               N/A


OUTPUT
FILES(S):               1) .PNG/.PDF/.EPS file that shows the depletion layer 
                        thickness as a function of mu_bar.



INPUT
ARGUMENT(S):            N/A
    
    
CREATED:                22Nov22

MODIFICATIONS
LOG:
22Nov22                 Migrated code to generate the plots from the original 
                        script to its own instance here.
15Jun23                 Added additional axes to snow snapshots of filament 
                        behavior near the wall.

    
            
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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})

def ycom_traj_vault_snp_depl_layer(ensemble_avg_data_df,ld_fit_df,parameter_dict,
                                   loc_data_1,loc_data_2,time_data_1,time_data_2,
                                  data_type,file_name,output_directory):     
    """
    This function plots the following 3 things: 1) Instantaneous average y-
    center of mass as a function of time for 2 mu_bar values; 2) Snapshots that
    show the filament behavior near the wall for these 2 mu_bar values; and 3)
    scaling relationship between the depletion layer and mu_bar.
    
    Inputs:
        
    ensemble_avg_data_df:           Pandas DataFrame that contains the average
                                    y-center of mass trajectory based on the 
                                    initial filament starting position.
    ld_fit_df:                      Pandas dataframe that contains information
                                    regarding the scaling relationship between
                                    depletion layer and mu_bar.
    Parameter_dict:                 Dictionary that has the information regarding
                                    Poiseuille flow parameters and which mu_bar
                                    values to plot the data for.
    loc_data_1:                     Numpy array that contains the filament
                                    positional data at every timepoint for one
                                    mu_bar value.
    loc_data_2:                     Numpy array that contains the filament
                                    positional data at every timepoint for one
                                    mu_bar value.
    time_data_1:                    Numpy array that contains the simulation time
                                    values for one mu_bar value.
    time_data_2:                    Numpy array that contains the simulation time
                                    values for one mu_bar value.
    data_type:                      String to tell what kind of depletion layer
                                    data to plot (average ensemble vs all 
                                    ensemble). 
    file_name:                      Name of the resulting .PNG/.PDF/.EPS files.
    output_directory:               Directory to save the data in.
    """
    ### Set up Figure ###
    fig = plt.figure(figsize=(7,10))
    gs = gridspec.GridSpec(5, 5)
    
    # Trajectory plot axes
    ax0 = fig.add_subplot(gs[0,0:2])
    ax1 = fig.add_subplot(gs[0,3:5])
    
    # Vaulting snapshot axes
    subplots_grid = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[1:3,:],hspace=0.00,wspace = 0)
    
    ### Store all of the subplot grid axes in a list ###
    all_subplot_grid_axes = []
    all_subplot_grid_r1 = []
    all_subplot_grid_r2 = []
    for i in range(2):
        for j in range(5):
            ax = plt.subplot(subplots_grid[i, j])
            if i == 0:
                all_subplot_grid_r1.append(ax)
            elif i == 1:
                all_subplot_grid_r2.append(ax)
    all_subplot_grid_axes.append(all_subplot_grid_r1)
    all_subplot_grid_axes.append(all_subplot_grid_r2)
    
    # Depletion Layer axes
    ax4 = fig.add_subplot(gs[3:,:])

    ### Plot COM Trajectory Curve Suplots ###
    mu_bar1 = parameter_dict["Mu_bar to Plot 1"]
    mu_bar2 = parameter_dict["Mu_bar to Plot 2"]
    mu_bar_df1 = ensemble_avg_data_df[ensemble_avg_data_df["Mu_bar"] == mu_bar1]
    mu_bar_df2 = ensemble_avg_data_df[ensemble_avg_data_df["Mu_bar"] == mu_bar2]
        
    sns.lineplot(y = 'Center of Mass-y',x = 'Brownian Time',hue = 'Starting Vertical Displacement',
                            palette = 'tab20',data = mu_bar_df1,linewidth = 1.5,
                            legend = False,ax = ax0)
    sns.lineplot(y = 'Center of Mass-y',x = 'Brownian Time',hue = 'Starting Vertical Displacement',
                            palette = 'tab20',data = mu_bar_df2,linewidth = 1.5,
                            legend = False,ax = ax1)
    
    ### Format COM Trajectory curves ###
    for ax in [ax0,ax1]:
        ax.set_xlim(-1e-3,5.05e-2)
        ax.set_xticks(np.linspace(0,5e-2,6))
        ax.axhline(y = 0.5,xmin = 0,
                      xmax = 1,color = 'gray',alpha = 0.4,
                      linestyle = 'dashed')
        ax.set_ylim(-0.01,0.55)
        ax.set_yticks(np.linspace(-0,0.5,6))
        ax.tick_params(axis = 'both',which = 'both',direction = 'in',labelsize = 11,pad = 5)
        ax.set_aspect(np.diff(ax.get_xlim())/(1.5*np.diff(ax.get_ylim())))
        ax.set_xlabel(r"$t^{\text{Br}} \times 10^{-2}$",fontsize = 13,labelpad = 3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2,-2))
        ax.xaxis.offsetText.set_visible(False)
        ax.set_ylabel(r"$\langle y^{\text{com}}\rangle (t)$",fontsize = 13,labelpad = 5)
        
        
    # Annotate depletion layer in Trajectory curves
    ax0.annotate(text = "",xy = (4.8e-2,0.15),
                      xytext = (4.8e-2,0.5),arrowprops=dict(arrowstyle='<->', connectionstyle='arc3,rad=0', 
                                color='black',linewidth = 1.2,shrinkA=1, shrinkB=1))
    ax0.text(x = 3.9e-2,y=0.31,s = r"$\langle L_{d}\rangle $",size = 13)
    
    ax1.annotate(text = "",xy = (4.8e-2,0.28),
                      xytext = (4.8e-2,0.5),arrowprops=dict(arrowstyle='<->', connectionstyle='arc3,rad=0', 
                                color='black',linewidth = 1.2,shrinkA=1, shrinkB=1))
    ax1.text(x = 3.9e-2,y=0.40,s = r"$\langle L_{d}\rangle $",size = 13)
    
    ### Plot snapshots of filament vaulting ###
    timepoints_low_mu_bar = np.array([1e-3,1.2e-3,1.4e-3,1.45e-3,1.9e-3]) #Need 5 values
    timepoints_high_mu_bar = np.array([9e-5,1e-4,1.1e-4,1.2e-4,1.3e-4]) #Need 5 values
    
    for n_row,ax_row in enumerate(all_subplot_grid_axes):
        for n_col,ax_col in enumerate(ax_row):
            if n_row == 0:
                snp1_idx = np.where(time_data_1 == timepoints_low_mu_bar[n_col])[0][0]
                ax_col.plot(loc_data_1[:,0,snp1_idx],
                                                 loc_data_1[:,1,snp1_idx],color = '#C96480',linewidth = 2)
            elif n_row == 1:
                snp1_idx = np.where(time_data_2 == timepoints_high_mu_bar[n_col])[0][0]
                ax_col.plot(loc_data_2[:,0,snp1_idx],
                                                 loc_data_2[:,1,snp1_idx],color = '#C96480',linewidth = 2)
            
        
    ### Annotate to show progression of filament shape over time ###
    all_subplot_grid_axes[0][0].annotate(text = "",xy = (4.6,-0.60),
                      xytext = (-0.5,-0.60),arrowprops=dict(arrowstyle='->', connectionstyle='arc,rad=0', 
                                mutation_scale = 20,color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2),annotation_clip = False)
    all_subplot_grid_axes[1][0].annotate(text = "",xy = (4.6,-0.60),
                      xytext = (-0.5,-0.60),arrowprops=dict(arrowstyle='->', connectionstyle='arc,rad=0', 
                                mutation_scale = 20,color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2),annotation_clip = False)
    all_subplot_grid_axes[0][2].text(x = -0.15,y = -0.75,s = "Time",fontsize = 13)
    all_subplot_grid_axes[1][2].text(x = -0.15,y = -0.75,s = "Time",fontsize = 13)
    
    
    ### Plot Velocity profile in main subplot ###
    channel_height,u_centerline = parameter_dict['Channel Height'],parameter_dict['Poiseuille U Centerline']
    y = np.linspace(-1*0.5,1*channel_height,200)
    ux = (u_centerline*0.5*(1-(y**2)/(channel_height**2)))- 0.5
    all_subplot_grid_axes[0][0].plot(ux,y,color = 'blue',linewidth = 1.1,linestyle = 'solid',alpha = 0.2)
    all_subplot_grid_axes[1][0].plot(ux,y,color = 'blue',linewidth = 1.1,linestyle = 'solid',alpha = 0.2)
    slicing_factor = 20
    y_subset = y[::slicing_factor].copy()
    x_subset = ux[::slicing_factor].copy()
    quiv_x = np.zeros_like(y_subset) - 0.5
    quiv_y = y_subset.copy()
    quiv_ar_x = x_subset.copy() + 0.5
    quiv_ar_y = np.zeros_like(y_subset)
    all_subplot_grid_axes[0][0].quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                   scale_units='xy', scale=1,color = 'blue',headwidth = 12,
                   headlength = 10,alpha = 0.2)
    all_subplot_grid_axes[1][0].quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                   scale_units='xy', scale=1,color = 'blue',headwidth = 12,
                   headlength = 10,alpha = 0.2)
    
    ### Format snapshots of filament vaulting axes ###
    for n_row,ax_row in enumerate(all_subplot_grid_axes):
        for n_col,ax_col in enumerate(ax_row):
            ax_col.tick_params(axis='both', which='major', direction = 'in',labelsize=11,pad = 5)
            ax_col.set_ylim(-0.51,0.51)
            ax_col.set_yticks(np.linspace(-0.5,0.5,3))
            ax_col.set_xlim(-0.51,0.51)
            ax_col.set_xticks(np.linspace(-0.5,0.5,5))
            ax_col.get_xaxis().set_visible(False)
            
            if n_col == 0:
                # ax_col.axvline(x = -0.5,ymin = 0.5,ymax = 1,color = 'black',linewidth = 0.8)
                ax_col.set_ylabel(r"y",fontsize = 13,labelpad = 2)
                ax_col.spines['right'].set_visible(False)
            else:
                ax_col.get_yaxis().set_visible(False)
                ax_col.spines['left'].set_visible(False)
                if n_col != 4:
                    ax_col.spines['right'].set_visible(False)
            ax_col.set_aspect(np.diff(ax_col.get_xlim())/(np.diff(ax_col.get_ylim())))
        
    
    ### Plot Depletion Layer Scaling ###
    slope,intercept = parameter_dict["Fit Slope"],parameter_dict["Fit Intercept"]
    x = np.logspace(3,6,1000)
    y = intercept*x**(slope)
        
    sns.lineplot(x = 'Mu_bar',y = 'Distance From Wall',
                  data = ld_fit_df,
                  err_style="bars",
                  marker = 'o',markersize = 7,
                  linestyle = '',legend = False,
                  errorbar=("sd", 1),err_kws={'capsize':5,'capthick': 2},
                  color = '#1C7C54',ax = ax4)
    ax4.plot(x,y,linewidth = 1.2,linestyle = 'dashed',color = '#3A3335')
    

    ### Format Depletion Layer Scaling ###
    ax4.set(xscale="log", yscale="log")
    ax4.set_ylim(1.6e-1,4.3e-1)
    ax4.set_xlabel(r"$\bar{\mu}$",fontsize = 13,labelpad = 5)
    if data_type == 'average_com':
        # axes.set_ylabel(r"$H - \langle y^{\text{com}}_{f}\rangle$",fontsize = 13,labelpad = 5)
        ax4.set_ylabel(r"$\langle L_{d} \rangle $",fontsize = 13,labelpad = 5)
    elif data_type == 'ensemble_com':
        ax4.set_ylabel(r"$H - y^{\text{com}}_{f}$",fontsize = 13,labelpad = 5)
    ax4.tick_params(axis='both', which='major', direction = 'in',labelsize=13,pad = 5)
    ax4.tick_params(axis='both', which='minor', direction = 'in',labelsize=11,pad = 5)
    
    ### Label each subfigure ###
    ax0.text(x = -2.6e-2,y = 0.5,s = r"\textbf{(a)}",size = 15)
    ax1.text(x = -2.6e-2,y = 0.5,s = r"\textbf{(b)}",size = 15)
    all_subplot_grid_axes[0][0].text(x = -1.35,y = 0.55,s = r"\textbf{(c)}",size = 15)
    all_subplot_grid_axes[1][0].text(x = -1.35,y = 0.55,s = r"\textbf{(d)}",size = 15)
    ax4.text(x = 2e2,y = 4.1e-1,s = r"\textbf{(e)}",size = 15)
    
    filename_png = '{}.png'.format(file_name)
    filename_pdf = '{}.pdf'.format(file_name)
    filename_eps = '{}.eps'.format(file_name)
    plt.savefig(os.path.join(output_directory,filename_png),bbox_inches = 'tight',
                format = 'png',dpi = 600)
    plt.savefig(os.path.join(output_directory,filename_pdf),bbox_inches = 'tight',
                format = 'pdf',dpi = 600)
    plt.savefig(os.path.join(output_directory,filename_eps),bbox_inches = 'tight',
                dpi = 600,format = 'eps')
    plt.show()