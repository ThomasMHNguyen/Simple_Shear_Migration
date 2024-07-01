# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:10:13 2023

@author: super
"""
import os, sys
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import numpy as np
import seaborn as sns

### comment this line if running from terminal ###
current_dir = os.chdir('C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01a_Post_Processing_Videos\\')
sys.path.append(current_dir)
from animate import video_animation,new_frame_single_plot
#%%

def new_frame_sp(ind,animation1,fig,filament_data,time_data,flow_type,brownian):
    """
    This function updates the position/tension of the filament as well as the 
    time elapsed in the simulation on the title of the plot of the animation. 
    This function is intended to be used for dual animations.
    
    Inputs:
    ind:            Frame to be plotted in thje animation.
    animation:      Animation plot whose title is updated.
    ax:             Axis of plot.
    ani_type:       Type of animation to be produced.
    """
    
    rigidity_title = filament_data.rigidity_title
    position_filament = filament_data.position_data
    kflow_text,kflow_freq = filament_data.kflow_phase_text,filament_data.kflow_phase_val
    if flow_type == 'Shear':
        if brownian:
            fig.suptitle(r"$t^{{Br}} = {0:.3e} | U_{{x}} = y | \bar{{\mu}} = {3:.2e}$".format(time_data[ind],filament_data.mu_bar),fontsize = 13,y = 0.980)
        else:
            # fig.suptitle(r"$t = {0:.3e} | U_{{x}} = y | \bar{{\mu}} = {3:.2e}$".format(time_data[ind],filament_data.mu_bar),fontsize = 13,y = 0.980)
            fig.suptitle(
                r"$t = {0:.3e} $".format(time_data[ind]),
                fontsize = 21,y = 0.960)
    elif flow_type == 'Poiseuille':
        if brownian:
            fig.suptitle(
                r"$t^{{Br}} = {0:.3e} | U_{{x}} = {1}\left(1-\frac{{y^{{2}}}}{{{2}^{{2}}}}\right) | \bar{{\mu}} = {3:.2e}$".format(time_data[ind],
                                                                                                            filament_data.U_centerline,
                                                                                                            filament_data.channel_height,
                                                                                                            filament_data.mu_bar),
                fontsize = 13,y = 0.980)
            
            # fig.suptitle(
            #     r"$t^{{Br}} = {0:.3e} | U_{{x}} = 1-\frac{{y^{{2}}}}{{{1}^{{2}}}} | \bar{{\mu}} = {2:.2e}$".format(time_data[ind],
            #                                                                                                filament_data.channel_height,
            #                                                                                                filament_data.mu_bar),
            # fontsize = 13,y = 0.980)
            # fig.suptitle(
            #     r"$t^{{Br}} = {0:.3e} $".format(time_data[ind]),
            #     fontsize = 18,y = 0.960)
        else:
            # fig.suptitle(
            #     r"$t = {0:.3e} | U_{{x}} = {1}\left(1-\frac{{y^{{2}}}}{{{2}^{{2}}}}\right) | \bar{{\mu}} = {3:.2e}$".format(time_data[ind],
            #                                                                                         filament_data.U_centerline,
            #                                                                                         filament_data.channel_height,
            #                                                                                         filament_data.mu_bar),
            #     fontsize = 13,y = 0.980)
            fig.suptitle(
                r"$t^{{Br}} = {0:.3e} $".format(time_data[ind]),
                fontsize = 18,y = 0.960)
            
    elif flow_type == 'Kolmogorov':
        fig.suptitle(r"$U_{{x}} =$ sin$\left({0:.0f} \times {1:.0f} y\right) | \bar{{\mu}} = {3:.2e}$".format(kflow_text,kflow_freq,filament_data.mu_bar))
   
    animation1.set_data(position_filament[:,0,ind],position_filament[:,1,ind]) 