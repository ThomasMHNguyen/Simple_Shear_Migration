# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 23:32:11 2023

@author: super
"""
import os, sys, time, math, logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

### comment this line if running from terminal ###
current_dir = os.chdir('C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01a_Post_Processing_Videos\\')
sys.path.append(current_dir)

from animate import new_frame_double_plot ,new_frame_single_plot

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": "Times",
#     'text.latex.preamble': r'\usepackage{amsmath}'})


#%%

def filament_position(filament_data,flow_type,brownian,sterics,upper_channel_height,lower_channel_height):
    """
    This function simultaneously creates the animation for the filament position 
    and tension changing during the simulation.
    
    Inputs:
    filament_data:      Python class that contains all of the parameters and data
                        related to the filament simulaiton. 
    """
    
    position_filament,tension_filament = filament_data.position_data,filament_data.tension_data
    output_dir = filament_data.output_dir
    s,tot_frames = filament_data.s,filament_data.total_frames
    
    dynamic_color = "#9E2B25"
    initial_color = '#0D3B66'
    

    start_time = time.perf_counter()  
    fig,axes = plt.subplots(ncols = 1,figsize = (7,7))
    fig.subplots_adjust(left=None, bottom=None, right=0.9, top=0.9)
    axes.plot(position_filament[:,0,0],position_filament[:,1,0],color = initial_color,lw = 2.5,label = 'Initial Position')
    ani1, = axes.plot(position_filament[:,0,0],position_filament[:,1,0],color = dynamic_color,lw = 2.5,label = 'Dynamic Position')
    
    ### Plot Velocity profile in main subplot ###
    if flow_type == 'Poiseuille':
        offset = -0.55
        channel_height,u_centerline = upper_channel_height,filament_data.U_centerline
        y = np.linspace(-1*0.7,1*0.7,200)
        ux = (u_centerline*0.25*(1-(y**2)/(channel_height**2))) + offset
        axes.plot(ux,y,color = 'blue',linewidth = 1.1,linestyle = 'solid',alpha = 0.2)
        slicing_factor = 20
        y_subset = y[::slicing_factor].copy()
        x_subset = ux[::slicing_factor].copy()
        quiv_x = np.zeros_like(y_subset) +offset
        quiv_y = y_subset.copy()
        quiv_ar_x = x_subset.copy() - offset
        quiv_ar_y = np.zeros_like(y_subset)
        axes.quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                       scale_units='xy', scale=1,color = 'blue',alpha = 0.2)

    elif flow_type == 'Shear':
        offset = -0.50
        channel_height = upper_channel_height
        y = np.linspace(-1.5*channel_height,1.5*channel_height,200)
        ux = 0.2*y.copy() + offset
        axes.plot(ux,y,color = 'blue',linewidth = 1.2,
                     linestyle = 'solid',alpha = 0.2)
        axes.axvline(x = offset,ymin = 0,ymax = 1,
                       color = 'blue',linewidth = 1.2,
                       linestyle = 'solid',alpha = 0.2)
        slicing_factor = 20
        y_subset = y[::slicing_factor].copy()
        x_subset = ux[::slicing_factor].copy()
        quiv_x = np.zeros_like(y_subset) +offset
        quiv_y = y_subset.copy()
        quiv_ar_x = x_subset.copy() - offset
        quiv_ar_y = np.zeros_like(y_subset)
        axes.quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                       scale_units='xy', scale=1,color = 'blue',alpha = 0.2)
        
        
    #Format axes
    if sterics:
        if upper_channel_height:
            axes.axhline(y = upper_channel_height,xmin = 0,xmax = 1,color = 'black',linewidth = 1.2)
        if lower_channel_height:
            axes.axhline(y = lower_channel_height,xmin = 0,xmax = 1,color = 'black',linewidth = 1.2)
    axes.set_xlim(-0.55,0.55)
    axes.set_ylim(-0.55,0.55)
    # axes.set_xlim(-0.55,0.55)
    # axes.set_ylim(-0.55,0.55)
    axes.set_xticks(np.linspace(-0.5,0.5,5))
    axes.set_yticks(np.linspace(-0.5,0.5,5))
    axes.tick_params(axis='both', which='major',direction = 'in', labelsize=11)
    axes.axvline(x=0, color='gray', linestyle='dashed',linewidth = 1)
    axes.axhline(y=0, color='gray', linestyle='dashed',linewidth = 1)
    # axes.legend(loc='center left',prop={'size': 12},bbox_to_anchor = (0.57,0.13))
    axes.set_xlabel(r'$x$',fontsize = 13,labelpad = 5)
    axes.set_ylabel(r'$y$',fontsize = 13,labelpad = 5)
    axes.set_aspect(np.diff(axes.get_xlim())/np.diff(axes.get_ylim()))
    movie1 = animation.FuncAnimation(fig, new_frame_single_plot.new_frame_sp, 
                                      fargs = (ani1,fig,filament_data,filament_data.time_values,flow_type,brownian,),
                                      frames=tot_frames, \
                                          interval=0.75, blit=False,repeat=True)  
    movie1.save(os.path.join(output_dir,'paper_video_final.mp4'),
                fps = int(math.floor(tot_frames.shape[0]/30)), extra_args=['-vcodec', 'libx264'],dpi = 400)
    end_time = time.perf_counter()
    logging.info("Time to finish plotting filament position: {:.0f} seconds.".format(end_time - start_time))
    
    
def filament_position_tension(filament_data,flow_type,brownian,sterics,upper_channel_height,lower_channel_height):
    
    position_filament,tension_filament = filament_data.position_data,filament_data.tension_data
    output_dir = filament_data.output_dir
    s,tot_frames = filament_data.s,filament_data.total_frames
    end_idx = 0
    if end_idx == 0:
        end_label = r'$s = -1/2$'
    elif end_idx == -1:
        end_label = r'$s = +1/2$'
    
    #colors
    dynamic_color = "#9E2B25"
    initial_color = '#0D3B66'
    marker_color = '#353A47'
    
    start_time = time.perf_counter()  
    fig,axes = plt.subplots(ncols = 2,figsize = (12,7),layout = 'constrained')
    
    #Plot Initial Data
    axes[0].plot(position_filament[:,0,0],position_filament[:,1,0],color = initial_color,linestyle = 'dashed',lw = 1.5,label = 'Initial Position')
    axes[1].plot(s,tension_filament[:,0],color = initial_color,linestyle = 'dashed',lw = 1.5,label = 'Initial Tension')
    
    #Set up dynamic data
    ani1, = axes[0].plot(position_filament[:,0,0],position_filament[:,1,0],color = dynamic_color,lw = 1.5,label = 'Dynamic Position')
    ani2, = axes[1].plot(s,tension_filament[:,0],color = dynamic_color,lw = 1.5,label = 'Dynamic Tension')
    ani3, = axes[0].plot(position_filament[end_idx,0,0],position_filament[end_idx,1,0],linestyle = 'None',color = marker_color,marker = 'o',markersize = 4,label = end_label)
    
    
    ### Plot Velocity profile in main subplot ###
    if flow_type == 'Poiseuille':
        offset = -0.60
        channel_height,u_centerline = upper_channel_height,filament_data.U_centerline
        y = np.linspace(-1*0.7,1*0.7,200)
        ux = (u_centerline*0.25*(1-(y**2)/(channel_height**2))) + offset
        axes[0].plot(ux,y,color = 'blue',linewidth = 1.1,linestyle = 'solid',alpha = 0.2)
        slicing_factor = 20
        y_subset = y[::slicing_factor].copy()
        x_subset = ux[::slicing_factor].copy()
        quiv_x = np.zeros_like(y_subset) +offset
        quiv_y = y_subset.copy()
        quiv_ar_x = x_subset.copy() - offset
        quiv_ar_y = np.zeros_like(y_subset)
        axes[0].quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                       scale_units='xy', scale=1,color = 'blue',alpha = 0.2)

    elif flow_type == 'Shear':
        offset = -0.55
        channel_height = upper_channel_height
        y = np.linspace(-1.5*channel_height,1.5*channel_height,200)
        ux = 0.2*y.copy() + offset
        axes[0].plot(ux,y,color = 'blue',linewidth = 1.2,
                     linestyle = 'solid',alpha = 0.2)
        axes[0].axvline(x = offset,ymin = 0,ymax = 1,
                       color = 'blue',linewidth = 1.2,
                       linestyle = 'solid',alpha = 0.2)
        slicing_factor = 20
        y_subset = y[::slicing_factor].copy()
        x_subset = ux[::slicing_factor].copy()
        quiv_x = np.zeros_like(y_subset) +offset
        quiv_y = y_subset.copy()
        quiv_ar_x = x_subset.copy() - offset
        quiv_ar_y = np.zeros_like(y_subset)
        axes[0].quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                       scale_units='xy', scale=1,color = 'blue',alpha = 0.2)
    
    #Format position animation
    if sterics:
        if upper_channel_height:
            axes.axhline(y = upper_channel_height,xmin = 0,xmax = 1,color = 'black',linewidth = 1.2)
        if lower_channel_height:
            axes.axhline(y = lower_channel_height,xmin = 0,xmax = 1,color = 'black',linewidth = 1.2)
    axes[0].set_xlim(-0.6,0.6)
    axes[0].set_ylim(-0.6,0.6)
    axes[0].set_xticks(np.linspace(-0.5,0.5,5))
    axes[0].set_yticks(np.linspace(-0.5,0.5,5))
    axes[0].tick_params(axis='both', which='major',direction = 'in', labelsize=11)
    axes[0].axvline(x=0, color='gray', linestyle='dashed',linewidth = 1)
    axes[0].axhline(y=0, color='gray', linestyle='dashed',linewidth = 1)
    axes[0].legend(loc='lower right',prop={'size': 12})
    axes[0].set_xlabel(r'$x$',fontsize = 13,labelpad = 5)
    axes[0].set_ylabel(r'$y$',fontsize = 13,labelpad = 5)
    axes[0].set_aspect(np.diff(axes[0].get_xlim())/np.diff(axes[0].get_ylim()))
    axes[0].set_title("Filament Position",size = 13,pad = 3)
    
    #Format Tension animation
    axes[1].set_xlabel(r'$s$',fontsize = 13,labelpad = 5)
    axes[1].set_ylabel(r'$T(s)$',fontsize = 13,labelpad = 5)
    axes[1].set_ylim(-450,350)
    axes[1].set_xlim(-0.6,0.6)
    axes[1].set_xticks(np.linspace(-0.5,0.5,5))
    axes[1].set_yticks(np.linspace(-400,300,8))
    axes[1].tick_params(axis='both', which='both',direction = 'in',labelsize = 11)
    axes[1].axhline(y=0, color='gray', linestyle='dashed',linewidth = 1)
    axes[1].legend(loc='lower right',prop={'size': 12})
    axes[1].set_title("Filament Tension",size = 13,pad = 3)
    [l.set_visible(False) for (i,l) in enumerate(axes[1].yaxis.get_ticklabels()) if (i) % 2 != 0]
    
    
    axes[1].set_aspect(np.diff(axes[1].get_xlim())/np.diff(axes[1].get_ylim()))
    movie1 = animation.FuncAnimation(fig, new_frame_double_plot.new_frame_dp, 
                                      fargs = (ani1,ani2,ani3,fig,axes,filament_data,marker_color,end_idx,flow_type,brownian,),
                                      frames=tot_frames, \
                                          interval=0.75, blit=False,repeat=True)  
    movie1.save(os.path.join(output_dir,'filament_mov_8.mp4'),
                fps = int(math.floor(tot_frames.shape[0]/30)), extra_args=['-vcodec', 'libx264'],dpi = 600)
    end_time = time.perf_counter()
    print("Total computing time for position and tension simultaneous animation took {} seconds".format(np.round(end_time - start_time,2)))
