# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 17:37:00 2023

@author: super
"""

import re, os,argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_com_distribution_subplots(ensemble_data_df,mu_bar_1,mu_bar_2,time_vals,time_vals_text,output_dir):
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
            
            
            # ax_col.set_title(r"$t^{{\text{{Br}}}} = {0}$".format(time_vals_text[n_col]),
            #                     fontsize = 13,pad = 4)
            
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
    

def plot_com_distribution_org(ensemble_data_df,mu_bar_1,mu_bar_2,time_vals,time_vals_text,output_dir):
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
    
    y = np.linspace(-1*0.5,1*channel_height,200)
    ux = u_centerline*(1-(y**2)/(channel_height**2))
    
    fig = plt.figure(figsize=(10,7))
    gs = gridspec.GridSpec(2, 6)

    ### Create the main subplot to the left
    main_subplot_ax = plt.subplot(gs[:,:2])  # Spanning all rows, first two columns

    ### Create the subplots grid
    subplots_grid = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[:, 2:], wspace=1, hspace=0.05)
    
    ### Plot Velocity profile in main subplot ###
    main_subplot_ax.plot(ux,y,color = 'blue',linewidth = 1.5,linestyle = 'solid')
    slicing_factor = 10
    y_subset = y[::slicing_factor].copy()
    x_subset = ux[::slicing_factor].copy()
    quiv_x = np.zeros_like(y_subset)
    quiv_y = y_subset.copy()
    quiv_ar_x = x_subset.copy()
    quiv_ar_y = np.zeros_like(y_subset)
    main_subplot_ax.quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                   scale_units='xy', scale=1,color = 'blue',headwidth = 15,
                   headlength = 7)
    
    
    ### Format main subplot ###
    
    main_subplot_ax.set_xlabel(r"$U_{x} (y)$",
                        fontsize = 13,labelpad = 5)                  
    main_subplot_ax.set_ylabel(r"$y$",fontsize = 13,labelpad = 5)
    main_subplot_ax.set_ylim(-0.05,np.round(1.1*channel_height,2))
    main_subplot_ax.set_yticks(np.linspace(0,0.5,6))
    main_subplot_ax.set_xlim(-0.1,1.2)
    main_subplot_ax.set_xticks(np.linspace(-0,1,5))
    main_subplot_ax.hlines(y = channel_height,xmin =-0.1,xmax = 1.6,
             color = 'black',linestyle = 'dashed',linewidth = 1.2,alpha = 0.7)
    [l.set_visible(False) for (i,l) in enumerate(main_subplot_ax.xaxis.get_ticklabels()) if (i) % 2 != 0]
    main_subplot_ax.tick_params(axis='both', which='major',direction = 'in', labelsize=11,pad = 5)
    main_subplot_ax.vlines(x = 0,ymin = main_subplot_ax.get_ylim()[0],ymax = channel_height,color = 'gray',alpha = 0.7)


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
            
    ### Plot COM Distribution in the subplots grid ###
    for n_row,ax_row in enumerate(all_subplot_grid_axes):
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
    
            # Format each subplot in the subplot grid #
            ax_col.set_ylabel(r"$y^{\text{com}}$",fontsize = 13,labelpad = 5)
            ax_col.set_xlabel(r"Density",fontsize = 13,labelpad = 1)
            ax_col.set_ylim(-0.05,np.round(1.1*channel_height,2))
            ax_col.set_xlim(0,13.3)
            ax_col.set_yticks(np.linspace(0,0.5,6))
            ax_col.set_xticks(np.linspace(0,12,5))
            ax_col.tick_params(axis='both', which='major',direction = 'in', labelsize=11,pad = 5)
            [l.set_visible(False) for (i,l) in enumerate(ax_col.xaxis.get_ticklabels()) if (i) % 2 != 0]
            # ax_col.set_title(r"$t^{{\text{{Br}}}} = {0}$".format(time_vals_text[n_col]),
            #                     fontsize = 13,pad = 4)
            ax_col.set_title(r"$t^{{\text{{Br}}}} = {0:.2f}$".format(time_vals[n_col]),
                                fontsize = 13,pad = 4)
            ax_col.set_aspect(70*np.diff(ax_col.get_xlim())/np.diff(ax_col.get_xlim()))
            
    ### Get size of subplot position to scale main subplot ###
    bbox = ax_col.get_position()
    subplot_width = bbox.width
    subplot_height = bbox.height 
    
    ### Scale Velocity distribution by same as subplots ###
    main_subplot_ax.set_aspect(2*np.diff(main_subplot_ax.get_xlim())/np.diff(main_subplot_ax.get_ylim()))
    main_subplot_ax.set_position([0.16, 0.14, 2.5*subplot_width, 2.5*subplot_height])  # [left, bottom, width, height]
    
    ### text to denote subfigure ###
    main_subplot_ax.text(x = -0.56,y = 0.55,s = r'\textbf{(a)}',size = 15)
    all_subplot_grid_axes[0][0].text(x = -14,y = 0.55,s = r'\textbf{(b)}',size = 15)
    all_subplot_grid_axes[1][0].text(x = -14,y = 0.55,s = r'\textbf{(c)}',size = 15)
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
                               