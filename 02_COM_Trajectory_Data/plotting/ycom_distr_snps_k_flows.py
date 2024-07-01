# -*- coding: utf-8 -*-
"""
FILE NAME:              ycom_distr_snps_k_flows.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):                Process_COM_Migration_Data.py


DESCRIPTION:            This script will plot the probability distribution of all filament
                        ensembles' y-COM at various time points in Kolmogorov flow. 


INPUT
FILES(S):               N/A

OUTPUT
FILES(S):               1) .PNG/.PDF/.EPS file that shows the probability distribution 
                        of the y-COM at various timepoints.



INPUT
ARGUMENT(S):            N/A

CREATED:                22Jan23

MODIFICATIONS
LOG:
    
22Jun23                 Extra subplot to the velocity and velocity gradient 
                        .profile


    
            
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

   
def ycom_distr_snps_k_flows(ensemble_data_df,time_vals,time_vals_text,output_dir):
    """
    This function creates a series of subplots that show: 1) the velocity and
    velocity gradient profile of Kolmogorov flow; 2) the y-center of mass distribution
    of all ensemble simulations at various points in time.
    
    Inputs:
        
    ensemble_data_df:               Pandas DataFrame that contains all ensemble
                                    data for Kolmogorov flow.
    time_vals:                      Numpy array of 5 time values to plot the 
                                    y-center of mass distribution data for.
    time_vals_text:                 List of 5 time values in text form that will
                                    be shown on the plot.
    output_dir:                     Directory where the plots will be saved in.
    """
    
    #group data by various parameters
    exp_groups = ensemble_data_df.groupby(
        by = ['Rigidity Suffix','Mu_bar','Channel Height', #0-2
              'Poiseuille U Centerline','Kolmogorov Phase Text','Kolmogorov Phase Value','Kolmogorov Frequency', #3-6
              'Steric Velocity Exponential Coefficient','Steric Velocity Gap Criteria', #7-8
              'Sterics Use','Flow Type']) #9-10
    for group in exp_groups.groups.keys():
        group_df = exp_groups.get_group(group)
        rigid,mu_bar,channel_h,u_centerline,k_phase_text,k_phase_val,k_freq = group[:7]
        steric_use,flow_type = group[-2:]
        
        time_val_0_data = group_df[group_df['Brownian Time'] == time_vals[0]]
        time_val_1_data = group_df[group_df['Brownian Time'] == time_vals[1]]
        time_val_2_data = group_df[group_df['Brownian Time'] == time_vals[2]]
        time_val_3_data = group_df[group_df['Brownian Time'] == time_vals[3]]
        time_val_4_data = group_df[group_df['Brownian Time'] == time_vals[4]]
        
        
        str_mu_bar_v1 = '{:.1e}'.format(mu_bar).split('e')
        if len(str_mu_bar_v1[0].split('.')[1]) == 1:
            pref = '{0}p{1}{2}'.format(str_mu_bar_v1[0].split('.')[0],'0',
                                       str_mu_bar_v1[0].split('.')[1])
        else:
            pref = '{0}p{1}'.format(str_mu_bar_v1[0].split('.')[0],
                                       str_mu_bar_v1[0].split('.')[1])
        mu_bar_text = '{0}{1}'.format(pref,str_mu_bar_v1[1].replace('+','e'))
        U_c_text = '{:.2f}'.format(u_centerline).replace('.','p')
        channel_h_text = '{:.2f}'.format(channel_h).replace('.','p')
        
        
        ### Set up Figure ###
        fig = plt.figure(figsize=(11,7))
        gs = gridspec.GridSpec(1, 7)
        
        #Velocity Profile plot axes
        ax0 = fig.add_subplot(gs[0,0])
        
        #COM Snapshots axes 
        subplots_grid = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0,2:],hspace=-0.2,wspace = 0.0)
        
        ### Store all of the subplot grid axes in a list ###
        all_subplot_grid_axes = []
        for i in range(5):
            ax = plt.subplot(subplots_grid[i])
            all_subplot_grid_axes.append(ax)
        
        
        ##### Plot Kolmogorov flows-COM #####
        if k_phase_text != 'Pi':
            flow_profile = r"U_{{x}} =$ sin$\left({0:.0f} \times {1:.2f} y\right)".format(k_freq,k_phase_text)
        else:
            flow_profile = r"U_{{x}} =$ sin$\left({0:.0f} \times \pi y\right)".format(k_freq)
        abr_flow_txt = 'KMG'
        filename_prefix = '{}_ST_{}_MB_{}_H_{}_FR_{}_PH_{}'.format(steric_use,
                                                                   abr_flow_txt,
                                                                   mu_bar_text,
                                                                   channel_h_text,
                                                                   int(k_freq),
                                                                   k_phase_text)
        sns.histplot(data = time_val_0_data,
                        y = 'Center of Mass-y',stat = 'density',
                        fill = 'True',alpha = 0.2,linewidth = 0.0,
                        # binwidth = (time_val_0_data['Center of Mass-y'].max() - time_val_0_data['Center of Mass-y'].min())/\
                        #     time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                        bins = time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                        kde = True,
                        ax = all_subplot_grid_axes[0],legend = False,color = "#7209B7")
                            
        
        sns.histplot(data = time_val_1_data,
                    y = 'Center of Mass-y',stat = 'density',
                    fill = 'True',alpha = 0.2,linewidth = 0,
                    binwidth = (time_val_0_data['Center of Mass-y'].max() - time_val_0_data['Center of Mass-y'].min())/\
                        time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                    # binwidth = 0.12,
                    kde = True,
                    ax = all_subplot_grid_axes[1],legend = False,color = "#7209B7")
        sns.histplot(data = time_val_2_data,
                    y = 'Center of Mass-y',stat = 'density',
                    fill = 'True',alpha = 0.2,linewidth = 0,
                    binwidth = (time_val_0_data['Center of Mass-y'].max() - time_val_0_data['Center of Mass-y'].min())/\
                        time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                    # binwidth = 0.12,
                    kde = True,
                    ax = all_subplot_grid_axes[2],legend = False,color = "#7209B7")
        sns.histplot(data = time_val_3_data,
                    y = 'Center of Mass-y',stat = 'density',
                    fill = 'True',alpha = 0.2,linewidth = 0,
                    binwidth = (time_val_0_data['Center of Mass-y'].max() - time_val_0_data['Center of Mass-y'].min())/\
                        time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                    # binwidth = 0.12,
                    kde = True,
                    ax = all_subplot_grid_axes[3],legend = False,color = "#7209B7")
        sns.histplot(data = time_val_4_data,
                    y = 'Center of Mass-y',stat = 'density',
                    fill = 'True',alpha = 0.2,linewidth = 0,
                    binwidth = (time_val_0_data['Center of Mass-y'].max() - time_val_0_data['Center of Mass-y'].min())/\
                        time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                    # binwidth = 0.12,
                    kde = True,
                    ax = all_subplot_grid_axes[4],legend = False,color = "#7209B7")
        
        
        ### Flow profile ###
        y = np.linspace(-2.1,3.1,200)
        x = np.sin(k_freq*k_phase_val*y)
        dx_dy = k_freq*k_phase_val*np.cos(k_freq*k_phase_val*y)
            
            
        ### Plot Velocity Profiles & Starting Points ###
        ax0.plot(x,y,color = 'blue',linewidth = 1.5,linestyle = 'solid',
                        label = r'$U_{x}$')
        
        ax0.plot(dx_dy,y,color = 'black',linewidth = 1.3,linestyle = 'dashed',
                        label = r'$\partial U_{x}/ \partial y$',alpha = 0.7)

        slicing_factor = 10
        y_subset = y[::slicing_factor].copy()
        x_subset = x[::slicing_factor].copy()
        quiv_x = np.zeros_like(y_subset)
        quiv_y = y_subset.copy()
        quiv_ar_x = x_subset.copy()
        quiv_ar_y = np.zeros_like(y_subset)
        ax0.quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                       scale_units='xy', scale=1,color = 'blue',headwidth = 15,
                       headlength = 7)
        
        subfigure_text = [r"\textbf{(a)}",r"\textbf{(b)}",r"\textbf{(c)}",r"\textbf{(d)}",r"\textbf{(e)}",r"\textbf{(f)}"]
        
        ### Format velocity profile plot
        ax0.text(x = -7.65,y = 2.5,s = subfigure_text[0],fontsize = 14)
        ax0.set_xlabel(r"$U_{x} (y) \text{ or } \partial U_{x}/ \partial y (y)$",
                            fontsize = 13,labelpad = 3)
        ax0.set_ylabel(r"$y$",fontsize = 13,labelpad = 2)
        ax0.set_ylim(-0.6,2.6)
        ax0.set_yticks(np.linspace(-0.5,2.5,7))
        ax0.set_xlim(-3.5,3.5)
        ax0.set_xticks(np.linspace(-3,3,7))
        ax0.tick_params(axis='both', which='major',direction = 'in', labelsize=11,pad = 5)
        [l.set_visible(False) for (i,l) in enumerate(ax0.xaxis.get_ticklabels()) if (i-1) % 2 != 0]
        # ax.legend(loc = 'center left',bbox_to_anchor = (-0.03,1.14),fontsize = 10)
        ax0.set_aspect(2*np.diff(ax0.get_xlim())/np.diff(ax0.get_ylim()))
        ax0.tick_params(axis='both', which='major',direction = 'in', labelsize=11,pad = 5)
        ax0.set_aspect(2.35*np.diff(ax0.get_xlim())/np.diff(ax0.get_ylim()))
        ax0.axvline(x = 0,ymin = 0,ymax = 1,color = 'gray',alpha = 0.7)
    
        
        ### Formatting on plots ###
        for n,ax in enumerate(all_subplot_grid_axes):
                ax.tick_params(axis='both', which='major',direction = 'in', labelsize=11,pad = 5)
                ax.set_ylim(-0.6,2.6)
                ax.set_xlim(0,1.2)
                x_ticks = [0,0.25,0.5,0.75,1]
                ax.set_xticks(np.linspace(0,1,5))
                ax.set_xticklabels(map(str,x_ticks))
                ax.text(x = 0.05,y = 2.7,s = subfigure_text[1:][n],fontsize = 14)
                if n == 0:
                    ax.set_ylabel(r"$y^{\text{com}}$",fontsize = 13,labelpad = 2)
                    ax.set_title(r"${0:.2f}$".format(time_vals[n]),
                                        fontsize = 11,pad = 5)
                    ax.set_yticks(np.linspace(-0.5,2.5,7))                     
                elif n > 0:
                    ax.get_yaxis().set_ticklabels([])
                    ax.set_ylabel("")
                    if n  == 1:
                        ax.set_title(r"${0:.3f}$".format(time_vals[n]),
                                            fontsize = 11,pad = 5)
                    else:
                        ax.set_title(r"${0:.2f}$".format(time_vals[n]),
                                            fontsize = 11,pad = 5)
                if n == 2:
                    ax.set_xlabel(r"Density",fontsize = 13,labelpad = 5)
                else:
                    ax.set_xlabel("")
                [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i) % 2 != 0]
                ax.set_aspect(2*np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
                        
        ### Save figure-COM ###
            
        filename_png = '{}.png'.format(filename_prefix)
        filename_pdf = '{}.pdf'.format(filename_prefix)
        filename_eps = '{}.eps'.format(filename_prefix)
        
        fig.savefig(os.path.join(output_dir,filename_png),bbox_inches = 'tight',
                    dpi = 600)
        fig.savefig(os.path.join(output_dir,filename_pdf),bbox_inches = 'tight',
                    dpi = 600)
        fig.savefig(os.path.join(output_dir,filename_eps),bbox_inches = 'tight',
                    dpi = 600,format = 'eps')
        plt.show()