# -*- coding: utf-8 -*-
"""
FILE NAME:      com_xs_shear_up.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        Compare_Poiseuille_Shear_Data.py


DESCRIPTION:    This script will read in the filament position data and simulation
                parameters performing an upward flip in Shear flow. Based on 
                the position data, it will plot theabsolute value of the 
                center of mass as a function of time. It will also plot the 
                first spatial derivative to approximate the stages of the 
                transition stages of the center of mass curve. 

INPUT
FILES(S):       N/A

OUTPUT
FILES(S):       

1)              .PDF/.SVG file that shows center of mass curves as a function of
                time for an upward/downward flip in Shear/Poiseuille flow. A curve
                that represents the maximum value of the absolute value of the 
                y-component of the first derivative will be superimposed on
                this graph as well.

INPUT
ARGUMENT(S):    N/A

CREATED:        19Jun23

MODIFICATIONS
LOG:            N/A
    
            
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.9.13

VERSION:        1.0

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:     N/A

NOTE(S):        
    
1)              The dependency of the he plotting routine is incased in the 
                poi_shear_data() class itself; it will import the plotting 
                function everytime the plotting routine is called.
2)              The resulting plot from this script has transparent components,
                which is supposedly unsupported by .PNG/.TIFF/.EPS files; thus,
                save the plots. as .PDF/.SVG .

"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
    'text.latex.preamble': r'\usepackage{amsmath}'})

    
def plot_ycom_xs_shear_up(time_data,position_data,center_of_mass_data,xs_data,\
                           xs_thres,file_name,output_dir):
    
    # Demarcate when to draw vertical lines for Phase II
    j_loop_idx = np.where(xs_data >= xs_thres)[0]
    j_loop_start,j_loop_end = j_loop_idx[0],j_loop_idx[-1]
    
    fig,axes = plt.subplots(figsize = (10,7),layout = 'constrained')
    
    # Inset plots
    ax1 = axes.twinx()
    axins_0 = inset_axes(axes, width="70%", height="70%",loc = 'upper left',
                         bbox_to_anchor = (100,250,200,200))
    axins_1 = inset_axes(axes, width="70%", height="70%",loc = 'upper left',
                         bbox_to_anchor = (280,250,200,200))
    axins_2 = inset_axes(axes, width="70%", height="70%",loc = 'upper right',
                         bbox_to_anchor = (425,250,200,200))
    
    
    ### Plot data and vertical lines ###
    axes.plot(time_data,center_of_mass_data,color = '#314CB6',linewidth = 2.5)
    ax1.plot(time_data,xs_data,color = '#8C1C13',linewidth = 2)
    ax1.vlines(x = time_data[j_loop_start],ymin = -1,ymax = 1.2,
               color = 'black',linestyle = 'dashed',linewidth = 0.8)
    ax1.vlines(x = time_data[j_loop_end],ymin = -1,ymax = 1.2,
               color = 'black',linestyle = 'dashed',linewidth = 0.8)
    
    
    ### Plot filament snapshots in inset ###
    snps_t1 = 1250
    snps_t2 = 1800
    snps_t3 = 2400
    axins_0.plot(position_data[:,0,snps_t1],position_data[:,1,snps_t1],
                 color = 'black',linewidth = 2)
    axins_1.plot(position_data[:,0,snps_t2],position_data[:,1,snps_t2],
                 color = 'black',linewidth = 2)
    axins_2.plot(position_data[:,0,snps_t3],position_data[:,1,snps_t3],
                 color = 'black',linewidth = 2)
    
    axins_0.plot(position_data[0,0,snps_t1],position_data[0,1,snps_t1],
                 color = 'purple',marker = 'd',markersize = 6)
    axins_1.plot(position_data[0,0,snps_t2],position_data[0,1,snps_t2],
                 color = 'purple',marker = 'd',markersize = 6)
    axins_2.plot(position_data[0,0,snps_t3],position_data[0,1,snps_t3],
                 color = 'purple',marker = 'd',markersize = 6)
    
    ### Plot flow profile in inset plots ###
    inset_axes_all = [axins_0,axins_1,axins_2]
    offset_vals = [-0.15,-0.15,-0.20]
    for idx,axins in enumerate(inset_axes_all):
        offset = offset_vals[idx]
        channel_height = 0.50
        y = np.linspace(-1.8*channel_height,1.8*channel_height,401) - 0.30
        ux = 0.7*y.copy() + offset + 0.30
        axins.plot(ux,y,color = '#262626',linewidth = 1.2,
                     linestyle = 'solid',alpha = 0.5)
        axins.axvline(x = offset,ymin = 0.015,ymax = 0.94,
                       color = '#262626',linewidth = 1.2,
                       linestyle = 'solid',alpha = 0.5)
        slicing_factor = 40
        y_subset = y[::slicing_factor].copy()
        x_subset = ux[::slicing_factor].copy()
        quiv_x = np.zeros_like(y_subset) +offset
        quiv_y = y_subset.copy()
        quiv_ar_x = x_subset.copy() - offset
        quiv_ar_y = np.zeros_like(y_subset)
        axins.quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                        scale_units='xy', scale=1,color = '#262626',
                        alpha = 0.6)
        # axins_1.quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
        #                scale_units='xy', scale=1,color = 'gray',
        #                alpha = 1,linewidths = 10*np.ones(6))
        # axins_2.quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
        #                 scale_units='xy', scale=1,color = 'gray',
        #                 alpha = 1,linewidth = 1.2)
    
    
    ### Format axes ###
    axes.set_ylabel(r"$\lvert y^{\text{com}} (t) - y_{0}\rvert $",fontsize = 21,labelpad = 7)
    axes.tick_params(axis = 'both',which = 'both',direction = 'in',width = 1.2,
                     length = 5,labelsize=20,pad = 3)
    axes.set_xlabel(r"$t$",fontsize = 21,labelpad = 5)
    axes.set_ylim(0.005,0.045)
    ax1.set_xlim(-1,41)
    axes.set_yticks(np.linspace(0.00,0.04,5))
    ax1.set_xticks(np.linspace(0,40,5))
    axes.set_aspect(np.diff(axes.get_xlim())/(1.5*np.diff(axes.get_ylim())))
    
    ax1.set_ylabel(r"$\text{max}\left(\lvert x_{s, y} (t) \rvert \right) $",fontsize = 21,labelpad = 7)
    ax1.set_ylim(-1e-4,1.05)
    ax1.set_xlim(-1,41)
    ax1.set_yticks(np.linspace(0.0,1,6))
    ax1.set_xticks(np.linspace(0,40,5))
    ax1.tick_params(axis = 'both',which = 'major',direction = 'in',width = 1.2,
                    length = 5,labelsize=20,pad = 3)
    ax1.set_aspect(np.diff(ax1.get_xlim())/(1.5*np.diff(ax1.get_ylim())))
    
    ### Format inset axes ###
    for ax in [axins_0,axins_1,axins_2]:
        ax.patch.set_alpha(0)
        ax.set_xlim(-0.68,0.68)
        ax.set_ylim(-0.68,0.68)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect(np.diff(ax.get_xlim())/(np.diff(ax.get_ylim())))
    
    ### Shade between the regions ###
    ax1.axvspan(-2,time_data[j_loop_start],color = "#EDEEC9",alpha = 0.3)
    ax1.axvspan(time_data[j_loop_start],time_data[j_loop_end],color = "#93827F",alpha = 0.3)
    ax1.axvspan(time_data[j_loop_end],53,color = "#558B6E",alpha = 0.3)
    
    ### Draw arrows to show direction of movement ###
    axins_0.annotate(text = "",xy = (-0.1,0.08),
                      xytext = (-0.4,0.10),
                      arrowprops=dict(arrowstyle='->', 
                                      connectionstyle='arc3,rad=-0.3', 
                                color='red',linewidth = 3,shrinkA=0.4, shrinkB=0.4))
    axins_1.annotate(text = "",xy = (-0.04,0.30),
                      xytext = (-0.25,0.14),
                      arrowprops=dict(arrowstyle='->',
                                      connectionstyle='arc3,rad=0.0', 
                                color='red',linewidth = 3,shrinkA=0.4, shrinkB=0.4))
    axins_1.annotate(text = "",xy = (-0.27,0.11),
                      xytext = (-0.15,-0.11),
                      arrowprops=dict(arrowstyle='->',
                                      connectionstyle='arc3,rad=-0.6', 
                                color='red',linewidth = 3,shrinkA=0.4, shrinkB=0.4))
    axins_1.annotate(text = "",xy = (-0.12,-0.13),
                      xytext = (0.14,-0.06),
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.0', 
                                color='red',linewidth = 3,shrinkA=0.4, shrinkB=0.4))
    axins_2.annotate(text = "",xy = (-0.43,-0.03),
                      xytext = (-0.32,-0.22),
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.5', 
                                color='red',linewidth = 3,shrinkA=0.4, shrinkB=0.4))
    
    ### Label transition points ###
    time_diff = time_data[1] - time_data[0]
    axes.text(x = j_loop_start*(time_diff)/2.1,y = 0.045,
              s = r"\textbf{I}",size = 21)
    axes.text(x = (j_loop_end-j_loop_start)*(time_diff)/2.1 + (j_loop_start*time_diff),
              y = 0.045,s = r"\textbf{II}",size = 21)
    axes.text(x = ((time_data[-1]/time_diff) - j_loop_end)*(time_diff)/2 + (j_loop_end*time_diff),
              y = 0.045,s = r"\textbf{III}",size = 21)
        
    pdf_filename = os.path.join(output_dir,'{}.pdf'.format(file_name))
    svg_filename = os.path.join(output_dir,'{}.svg'.format(file_name))
    

    plt.savefig(pdf_filename,
                format = 'pdf',bbox_inches = 'tight',dpi = 600)
    fig.savefig(svg_filename,
                bbox_inches = 'tight',format = 'svg',dpi = 600)
    
    plt.show()