# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:50:00 2023

@author: super
"""


import sys, os, logging
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd


def plot_time_course_panels(x_values_up,x_values_down,y_values_up,y_values_down,
                timepoints_up,timepoints_down,xlabels,ylabels,
                plot_type,output_dir,file_name):
    
    
    if timepoints_up.shape[0] == 81 and timepoints_up.shape[1] == 81:
        
        ### Determine plot axes limit ###
        if plot_type == 'position':
            xlim_l,xlim_h = - 0.6, 0.6
            xtick_l,xtick_h = -0.5,0.5
            xtick_n = 5
            ylim_l,ylim_h = - 0.6, 0.6
            ytick_l,ytick_h = -0.5,0.5
            ytick_n = 5
            
        elif plot_type == 'tension':
            xlim_l,xlim_h = - 0.6, 0.6
            xtick_l,xtick_h = -0.5,0.6
            xtick_n = 5
            ylim_l,ylim_h = -450,450
            ytick_l,ytick_h = -400,400
            ytick_n = 5
            
        elif plot_type == 'elastic_energy_time':
            xlim_l,xlim_h = -0.05,0.83
            xtick_l,xtick_h = 0,0.8
            xtick_n = 5
            ylim_l,ylim_h = -0.7,23
            ytick_l,ytick_h = 0,20
            ytick_n = 5
        elif plot_type == 'N1_Stress_time':
            xlim_l,xlim_h = -0.05,0.83
            xtick_l,xtick_h = 0,0.8
            xtick_n = 5
            ylim_l,ylim_h = -175,75
            ytick_l,ytick_h = -150,50
            ytick_n = 5
        elif plot_type == 'N2_Stress_time':
            xlim_l,xlim_h = -0.05,0.83
            xtick_l,xtick_h = 0,0.8
            xtick_n = 5
            ylim_l,ylim_h = -35,25
            ytick_l,ytick_h = -30,20
            ytick_n = 6
        elif plot_type == 'Sxy_Stress_time':
            xlim_l,xlim_h = -0.05,0.83
            xtick_l,xtick_h = 0,0.8
            xtick_n = 5
            ylim_l,ylim_h = -35,25
            ytick_l,ytick_h = -30,20
            ytick_n = 6
        elif plot_type == 'max_curvature_time':
            xlim_l,xlim_h = -0.05,0.83
            xtick_l,xtick_h = 0,0.8
            xtick_n = 5
            ylim_l,ylim_h = -2,17
            ytick_l,ytick_h = 0,15
            ytick_n = 4
        elif plot_type == 'end_angle_deg_time':
            xlim_l,xlim_h = -0.05,0.83
            xtick_l,xtick_h = 0,0.8
            xtick_n = 5
            ylim_l,ylim_h = -10,190
            ytick_l,ytick_h = 0,180
            ytick_n = 5
        elif plot_type == 'center_of_mass':
            xlim_l,xlim_h = -0.05,0.83
            xtick_l,xtick_h = 0,0.8
            xtick_n = 5
            ylim_l,ylim_h = 0,0.056
            ytick_l,ytick_h = 0,0.05
            ytick_n = 6
        elif plot_type == 'true_center':
            xlim_l,xlim_h = -0.05,0.83
            xtick_l,xtick_h = 0,0.8
            xtick_n = 5
            ylim_l,ylim_h = -0.012,0.012
            ytick_l,ytick_h = -0.10,0.10
            ytick_n = 5
        elif plot_type == 'center_of_mass_velocity':
            xlim_l,xlim_h = -0.05,0.83
            xtick_l,xtick_h = 0,0.8
            xtick_n = 5
            ylim_l,ylim_h = -0.001,0.013
            ytick_l,ytick_h = 0,0.01,6
            ytick_n = 5
        xtick_vals = np.linspace(xtick_l,xtick_h,xtick_n)
        ytick_vals = np.linspace(ytick_l,ytick_h,ytick_n)
        fig,axes = plt.subplots(nrows = 9,ncols = 9,
                                figsize = (13,13),layout = 'constrained',
                                sharey = True,sharex = True)
        time_counter = -1
        for n_row,ax_row in enumerate(axes):
            for n_col,ax_col in enumerate(ax_row):
                time_counter += 1
                curr_time_up_idx = np.where(x_values_up == timepoints_up[time_counter])[0][0]
                curr_time_down_idx = np.where(x_values_down == timepoints_down[time_counter])[0][0]
                
                if plot_type == 'position':
                    ax_col.plot(x_values_up[:,curr_time_up_idx],
                                y_values_up[:,curr_time_up_idx],
                                color = '#BCB6FF',label = 'Up')
                    ax_col.plot(x_values_down[:,curr_time_down_idx],
                                y_values_down[:,curr_time_down_idx],
                                color = '#B74F6F',label = 'Down') 
                    ax_col.plot(x_values_up[-1,curr_time_up_idx],
                                y_values_up[-1,curr_time_up_idx],
                                color = 'cyan',label = r'$s = 1/2$',marker = 'd',
                                markersize = 2)
                    ax_col.plot(x_values_down[0,curr_time_down_idx],
                                y_values_down[0,curr_time_down_idx],
                                color = 'magenta',label = r'$s = -1/2$',marker = 'd',
                                markersize = 2)
                elif plot_type == 'tension':
                    ax_col.plot(x_values_up,
                                y_values_up[:,curr_time_up_idx],
                                color = '#BCB6FF',label = 'Up')
                    ax_col.plot(x_values_down,
                                y_values_down[:,curr_time_down_idx],
                                color = '#B74F6F',label = 'Down') 
                else:
                    ### Plot Filament Data vs. Time ###
                    ax_col.plot(x_values_up,
                                y_values_up,
                                color = '#BCB6FF',label = 'Up')
                    ax_col.plot(x_values_down,
                                y_values_down,
                                color = '#B74F6F',label = 'Down')
                    
                    ax_col.plot(x_values_up[curr_time_up_idx],
                                y_values_up[curr_time_up_idx],
                                color = 'cyan',marker = 'o',markersize = 1.75)
                    ax_col.plot(x_values_down[curr_time_down_idx],
                                y_values_down[curr_time_down_idx],
                                color = 'magenta',marker = 'o',markersize = 1.75) 
                ax_col.set_xlim(xlim_l,xlim_h)
                ax_col.set_ylim(ylim_l,ylim_h)
                ax_col.set_xticks(xtick_vals)
                ax_col.set_yticks(ytick_vals)
                ax_col.tick_params(axis='both', which='major', labelsize=8)
                ax_col.set_aspect((ax_col.get_xlim()[1] -ax_col.get_xlim()[0])/\
                                                 (1*(ax_col.get_ylim()[1] - ax_col.get_ylim()[0])))
                ax_col.set_title(r"$t/t_{{\text{{max}}}} = {0:.2f}$".format(timepoints[time_counter]),
                color = 'black',size = 10,pad = 8)
                if n_col == 0:
                    ax_col.legend(loc = 'best',ncol = 2,prop={'size': 6})
                    
        fig.supxlabel(xlabels)
        fig.supylabel(ylabels)
        plt.savefig(os.path.join(output_dir,file_name),dpi = 400,bbox_inches = 'tight')    
        plt.show()
    else:
        logging.warning("The number of timepoints to plot does not match the number of subplots needed for this figure (81). Please respecify the timepoints to plot!")
        sys.exit(1)
        
        

