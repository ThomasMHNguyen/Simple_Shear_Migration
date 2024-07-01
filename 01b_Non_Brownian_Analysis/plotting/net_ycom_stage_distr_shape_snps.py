# -*- coding: utf-8 -*-
"""
FILE NAME:              net_ycom_stage_distr.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):                Compare_Poiseuille_Shear_Data.py

DESCRIPTION:            This script contains a function to plot representative motions
                        of filament movement (3 phases) for both upward and downward
                        flips in Poiseuille flow. Additionally, it will take in the
                        filament net drift data and plot the net displacement as based on
                        the movement phase as a strip plot (categorical scatter plot). 
                        The data points are color coded and shaped based on the direction 
                        of the flip. A color bar is also present.

INPUT
FILES(S):               N/A

OUTPUT
FILES(S):       
    
1)                      .PNG/.PDF/.EPS that contain: representative motions of filament flipping
                        phases in either the up or down direction in Poiseuille flow and a 
                        strip/swarm plot that shows the net drift of mass displacement curves 
                        based on each movement phase and flip direction. 

INPUT
ARGUMENT(S):            N/A
                

CREATED:                20Jun23

MODIFICATIONS
LOG:                    

18Jul23                 1) Added new function to plot filament shape snapshots as subplot
                        on Net drift violinplot. Swarmplots can now replace the violin
                        plots. 
01Aug23                 2) Added extra subplots to show the snapshots of each filament flipping 
                        direction.
    
            
LAST MODIFIED
BY:                     Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:                 3.9.13

VERSION:                1.1

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

### Use LaTeX to generate plots ###
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
    'text.latex.preamble': r'\usepackage{amsmath}'})


def net_ycom_stage_distr_shape_snps(input_file,up_poi_position_data,down_poi_position_data,
                                           output_directory,file_name):
    """
    This function plots representative motion of filament movement (3 phases)
    for both upward and downward flips in Poiseuille flow. Additionally, it will
    take in the filament net drift data and plot the net displacement as based on
    the movement phase as a strip plot (categorical scatter plot). The data points
    are color coded and shaped based on the direction of the flip. A color bar is also
    present.
    
    Inputs:
    
    input_file:                 Input Pandas dataframe that lists the measured
                                data values for each U-turn flipping event.
    up_poi_position_data:       Nx3XT array that contains the filament position
                                at every timepoint for an upward flip in 
                                Poiseuille flow.
    down_poi_position_data:     Nx3XT array that contains the filament position
                                at every timepoint for an downward flip in 
                                Poiseuille flow.
    output_directory:           Path to directory to be created.
    file_name:                  The file name to be used for the output graphical
                                files.
    """  
    fig= plt.figure(figsize = (7,10))
    gs = gridspec.GridSpec(4, 5)
    
    ### Filament shape snapshots ###
    subplots_grid = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0:2,:],hspace=0)
    
    # Store all of the subplot grid axes in a list #
    all_subplot_grid_axes = []
    all_subplot_grid_r1 = []
    all_subplot_grid_r2 = []
    for i in range(2):
        for j in range(3):
            ax = plt.subplot(subplots_grid[i, j])
            if i == 0:
                all_subplot_grid_r1.append(ax)
            elif i == 1:
                all_subplot_grid_r2.append(ax)
    all_subplot_grid_axes.append(all_subplot_grid_r1)
    all_subplot_grid_axes.append(all_subplot_grid_r2)
    all_subplot_grid_axes = np.array(all_subplot_grid_axes)

          
    # Snapshots per stage and flip direction, 3 different opacity values & colors
    opacity_vals = [0.30,0.50,0.95]
    colors_use = ["#F05D5E","#4D6CFA","#037971"]
    
    ### Upward Flips for each stage
    
    #Stage 1-Up
    timepoints_s1_up = [50,375,416]
    for i,v in enumerate(timepoints_s1_up):
        all_subplot_grid_axes[0,0].plot(up_poi_position_data[:,0,v],up_poi_position_data[:,1,v],color = colors_use[0],
                       linewidth = 2,alpha = opacity_vals[i])
        all_subplot_grid_axes[0,0].plot(up_poi_position_data[:,0,v],up_poi_position_data[:,1,v],color = colors_use[0],
                       linewidth = 2,alpha = opacity_vals[i])
    
    #Stage 2-Up
    timepoints_s2_up = [450,600,800]
    for i,v in enumerate(timepoints_s2_up):
        all_subplot_grid_axes[0,1].plot(up_poi_position_data[:,0,v],up_poi_position_data[:,1,v],color = colors_use[1],
                       linewidth = 2,alpha = opacity_vals[i])
        all_subplot_grid_axes[0,1].plot(up_poi_position_data[:,0,v],up_poi_position_data[:,1,v],color = colors_use[1],
                       linewidth = 2,alpha = opacity_vals[i])
    
    #Stage 3-Up
    timepoints_s3_up = [891,910,1000]
    for i,v in enumerate(timepoints_s3_up):
        all_subplot_grid_axes[0,2].plot(up_poi_position_data[:,0,v],up_poi_position_data[:,1,v],color = colors_use[2],
                       linewidth = 2,alpha = opacity_vals[i])
        all_subplot_grid_axes[0,2].plot(up_poi_position_data[:,0,v],up_poi_position_data[:,1,v],color = colors_use[2],
                       linewidth = 2,alpha = opacity_vals[i])
    
    ### Downward flips for each stage
    #Stage 1-Down
    timepoints_s1_down = [200,500,554]
    for i,v in enumerate(timepoints_s1_down):
        all_subplot_grid_axes[1,0].plot(down_poi_position_data[:,0,v],down_poi_position_data[:,1,v],color = colors_use[0],
                       linewidth = 2,alpha = opacity_vals[i])
        all_subplot_grid_axes[1,0].plot(down_poi_position_data[:,0,v],down_poi_position_data[:,1,v],color = colors_use[0],
                       linewidth = 2,alpha = opacity_vals[i])
    
    #Stage 2-Down
    timepoints_s2_down = [600,750,975]
    for i,v in enumerate(timepoints_s2_down):
        all_subplot_grid_axes[1,1].plot(down_poi_position_data[:,0,v],down_poi_position_data[:,1,v],color = colors_use[1],
                       linewidth = 2,alpha = opacity_vals[i])
        all_subplot_grid_axes[1,1].plot(down_poi_position_data[:,0,v],down_poi_position_data[:,1,v],color = colors_use[1],
                       linewidth = 2,alpha = opacity_vals[i])
    
    #Stage 3-Down
    timepoints_s3_down = [1044,1060,1200]
    for i,v in enumerate(timepoints_s3_down):
        all_subplot_grid_axes[1,2].plot(down_poi_position_data[:,0,v],down_poi_position_data[:,1,v],color = colors_use[2],
                       linewidth = 2,alpha = opacity_vals[i])
        all_subplot_grid_axes[1,2].plot(down_poi_position_data[:,0,v],down_poi_position_data[:,1,v],color = colors_use[2],
                       linewidth = 2,alpha = opacity_vals[i])
    
    ### Draw arrows that show direction of filament movement
    
    #Upward flips
    # axes[0,0].annotate(text = "",xy = (0.34,0.25),
    #                  xytext = (0.56,0.03),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.4', 
    #                             color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2))
    
    # axes[0,1].annotate(text = "",xy = (0.15,-0.15),
    #                  xytext = (-0.10,-0.05),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.0', 
    #                             color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2))
    # axes[0,1].annotate(text = "",xy = (0.22,0.10),
    #                   xytext = (0.20,-0.15),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.8', 
    #                             color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2))
    # axes[0,1].annotate(text = "",xy = (-0.05,0.18),
    #                   xytext = (0.18,0.12),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.0', 
    #                             color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2))
    
    # axes[0,2].annotate(text = "",xy = (0.55,-0.03),
    #                  xytext = (0.36,-0.24),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.4', 
    #                             color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2))
    
    #Downward flips
    # axes[1,0].annotate(text = "",xy = (-0.34,-0.27),
    #                  xytext = (-0.56,-0.03),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.4', 
    #                             color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2))
    
    # axes[1,1].annotate(text = "",xy = (-0.15,0.15),
    #                  xytext = (0.10,0.05),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.0', 
    #                             color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2))
    # axes[1,1].annotate(text = "",xy = (-0.19,-0.10),
    #                   xytext = (-0.16,0.15),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=1.1', 
    #                             color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2))
    # axes[1,1].annotate(text = "",xy = (0.13,-0.13),
    #                   xytext = (-0.09,-0.07),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.0', 
    #                             color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2))
    
    # axes[1,2].annotate(text = "",xy = (-0.55,0.03),
    #                  xytext = (-0.36,0.24),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.4', 
    #                             color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2))
    

    ### Format axes for filament position snapshots ###
    for n_row,ax_row in enumerate(all_subplot_grid_axes):
        for n_col,ax_col in enumerate(ax_row):
            offset = -0.55
            channel_height,u_centerline = 0.5,1
            y = np.linspace(-0.5,0.5,401)
            ux = (u_centerline*0.5*(1-(y**2)/(channel_height**2))) + offset
            ax_col.plot(ux,y,color = '#262626',linewidth = 1.1,linestyle = 'solid',alpha = 0.3)
            slicing_factor = 50
            y_subset = y[::slicing_factor].copy()
            x_subset = ux[::slicing_factor].copy()
            quiv_x = np.zeros_like(y_subset) +offset
            quiv_y = y_subset.copy()
            quiv_ar_x = x_subset.copy() - offset
            quiv_ar_y = np.zeros_like(y_subset)
            ax_col.quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                           scale_units='xy', scale=1,color = '#262626',alpha = 0.3)
            
            
            ax_col.axhline(y=0.30,xmin=0,xmax = 1, linewidth = 1.2,color = 'gray',linestyle = 'dashed')
            ax_col.set_ylim(-0.3,0.9)
            ax_col.set_yticks(np.linspace(-0.2,0.8,6))
            ax_col.set_xlim(-0.6,0.6)
            ax_col.set_xticks(np.linspace(-0.5,0.5,5))
            ax_col.spines['right'].set_visible(False)
            ax_col.spines['left'].set_visible(False)
            ax_col.spines['top'].set_visible(False)
            ax_col.spines['bottom'].set_visible(False)
            ax_col.get_xaxis().set_visible(False)
            ax_col.get_yaxis().set_visible(False)
            ax_col.set_aspect(np.diff(ax_col.get_xlim())/np.diff(ax_col.get_ylim()))
            
    ### Label each flip & stage ###
    all_subplot_grid_axes[0,0].text(x = -0.05,y = 0.78,s = r"I",size = 16)
    all_subplot_grid_axes[0,1].text(x = -0.05,y = 0.78,s = r"II",size = 16)
    all_subplot_grid_axes[0,2].text(x = -0.10,y = 0.78,s = r"III",size = 16)
    
    all_subplot_grid_axes[0,0].text(x = -0.90,y = 0.30,s = r"Up",size = 16)
    all_subplot_grid_axes[1,0].text(x = -1.10,y = 0.30,s = r"Down",size = 16)
    
    
    
    
    ### Strip plots that show difference in stages ###
    ax1 = fig.add_subplot(gs[2:,:])

    hue_order_fix = ["I","II","III"]
    vert_colorbar = 'inferno_r'
    # Plot Mean Data
    df_gb_flip = input_file.groupby(by = ['Displacement Type'])
    i = -1
    for group in df_gb_flip.groups.keys():
        group_df = df_gb_flip.get_group(group)
        if group == 'Down':
            g = sns.stripplot(x = 'Stage',y = 'Stage ABS Adjusted Center of Mass-y',
                                        data = group_df,hue = 'Starting Vertical Displacement',
                                        palette = vert_colorbar,dodge = True,jitter = 0.0,marker = 'd',
                                        order = hue_order_fix,size = 9,edgecolor = 'gray',
                                        alpha = 0.8,linewidth = 1.2,ax = ax1)
        elif group == 'Up':
            g = sns.stripplot(x = 'Stage',y = 'Stage ABS Adjusted Center of Mass-y',
                                        data = group_df,hue = 'Starting Vertical Displacement',
                                        palette = vert_colorbar,dodge = True,jitter = 0.0,marker = 'o',
                                        order = hue_order_fix,size = 9,edgecolor = 'gray',
                                        alpha = 0.8,linewidth = 1.2,ax = ax1)
        g.get_legend().remove()

    ### Create colorbar and formatting ###
    norm = plt.Normalize(input_file['Starting Vertical Displacement'].min(), input_file['Starting Vertical Displacement'].max())
    sm = plt.cm.ScalarMappable(cmap=vert_colorbar, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.12, 0.03, 0.78, 0.02])  # Adjust the position and size as needed (x0,y0,width,height)
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(0.25, 0.46, 0.05), orientation='horizontal')
    cbar.set_label(r"$y_{0}$", fontsize=16)
    cbar.ax.tick_params(labelsize=13)  # Specify the desired font size
    
    
    ### Format swarmplot ###
    ax1.tick_params(axis='both', which='both', direction = 'in',labelsize=14,pad = 5)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-3,-3))
    ax1.yaxis.offsetText.set_visible(False)
    ax1.set_ylim(5.8e-3,1.51e-2)
    ax1.set_yticks(np.linspace(6e-3,1.5e-2,4))
    ax1.set_xlabel(r"Movement Phase",fontsize = 16,labelpad = 5)
    ax1.set_ylabel(r"$\lvert \Delta y^{\text{com}}\rvert \times 10^{-3}$",fontsize = 16,labelpad = 5)
    ax1.set_aspect((np.diff(ax1.get_xlim()))/(1.5*np.diff(ax1.get_ylim())))
    
     # Black line to separate up & downward panels
    ax1.annotate(text = "",xy = (2.5,0.0198),
                      xytext = (-0.9,0.0198),arrowprops=dict(arrowstyle='-', connectionstyle='arc,rad=0', 
                                mutation_scale = 20,color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2),annotation_clip = False)
     
    ### Label each subfigure
    all_subplot_grid_axes[0,0].text(x = -1.28,y = 0.57,s = r"\textbf{(a)}",size = 17)
    ax1.text(x = -1.0,y = 1.56e-2,s = r"\textbf{(b)}",size = 17)
    
    fig.savefig(os.path.join(output_directory,'{}.png'.format(file_name)),
                bbox_inches = 'tight',dpi = 600)
    fig.savefig(os.path.join(output_directory,'{}.pdf'.format(file_name)),
                format = 'pdf',bbox_inches = 'tight',dpi = 600)
    fig.savefig(os.path.join(output_directory,'{}.eps'.format(file_name)),
                bbox_inches = 'tight',format = 'eps',dpi = 600)
    plt.show()