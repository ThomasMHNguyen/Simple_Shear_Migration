# -*- coding: utf-8 -*-
"""
FILE NAME:      com_xs_poi_up.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        Compare_Poiseuille_Shear_Data.py


DESCRIPTION:    This script will read in the filament position data and simulation
                parameters performing an upward flip in Poiseuille flow. 
                Based on the position data, it will plot the
                absolute value of the center of mass as a function of time.
                It will also plot the first spatial derivative to approximate 
                the stages of the transition stages of the center of mass curve. 

INPUT
FILES(S):       N/A

OUTPUT
FILES(S):       

1)              .PDF/.SVG file that shows center of mass curves as a function of
                time for an upward flip in Poiseuille flow. A curve
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
                save the plots. as .PDF/.SVG.

"""
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})



def plot_com_xs_poi_up(time_data,position_data,center_of_mass_data,xs_data,\
                           xs_thres,file_name,output_dir):
    
    # Demarcate where to plot the vertical bars
    j_loop_idx = np.where(xs_data >= xs_thres)[0]
    j_loop_start,j_loop_end = j_loop_idx[0],j_loop_idx[-1]
    print(j_loop_start,j_loop_end)
    
    fig,axes = plt.subplots(figsize = (10,7),layout = 'constrained')
    ax1 = axes.twinx()
    
    ### Set up inset axes to show snapshots of filament movement ###
    axins_0 = inset_axes(axes, width="70%", height="70%",loc = 'upper left',bbox_to_anchor = (53,250,200,200))
    axins_1 = inset_axes(axes, width="70%", height="70%",loc = 'upper left',bbox_to_anchor = (180,250,200,200))
    axins_2 = inset_axes(axes, width="70%", height="70%",loc = 'upper right',bbox_to_anchor = (360,250,200,200))
    
    
    ### Plot y_com curve & x_s curve
    axes.plot(time_data,center_of_mass_data,color = '#314CB6',linewidth = 2.5)
    ax1.plot(time_data,xs_data,color = '#8C1C13',linewidth = 2)
    ax1.vlines(x = time_data[j_loop_start],ymin = -1,ymax = 1.1,
                color = 'black',linestyle = 'dashed',linewidth = 0.8)
    ax1.vlines(x = time_data[j_loop_end],ymin = -1,ymax = 1.1,
                color = 'black',linestyle = 'dashed',linewidth = 0.8)
    
    ### Plot filament snapshots ###
    snps_t1 = 375
    snps_t2 = 500
    snps_t3 = 875
    axins_0.plot(position_data[:,0,snps_t1],position_data[:,1,snps_t1],color = 'black',linewidth = 2)
    axins_1.plot(position_data[:,0,snps_t2],position_data[:,1,snps_t2],color = 'black',linewidth = 2)
    axins_2.plot(position_data[:,0,snps_t3],position_data[:,1,snps_t3],color = 'black',linewidth = 2)
    
    
    ### Draw arrows to show filament direction ###
    axins_0.annotate(text = "",xy = (0.10,0.12),
                     xytext = (0.42,0.13),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                                color='red',linewidth = 2,shrinkA=0.4, shrinkB=0.4))
    axins_1.annotate(text = "",xy = (-0.01,0.28),
                      xytext = (0.25,0.13),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.0', 
                                color='red',linewidth = 2,shrinkA=0.4, shrinkB=0.4))
    axins_1.annotate(text = "",xy = (0.27,0.12),
                      xytext = (0.18,-0.12),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.8', 
                                color='red',linewidth = 2,shrinkA=0.4, shrinkB=0.4))
    axins_1.annotate(text = "",xy = (0.16,-0.15),
                      xytext = (-0.13,-0.07),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.0', 
                                color='red',linewidth = 2,shrinkA=0.4, shrinkB=0.4))
    axins_2.annotate(text = "",xy = (0.50,0.02),
                      xytext = (0.32,-0.20),arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                                color='red',linewidth = 2,shrinkA=0.4, shrinkB=0.4))
    
    ### Format axes ###
    axes.set_ylabel(r"$\vert y^{\text{com}}\: (t) - y_{0}\vert $",fontsize = 17,labelpad = 5)
    axes.tick_params(axis = 'both',which = 'both',direction = 'in',labelsize=15,pad = 3)
    axes.set_xlabel(r"$t$",fontsize = 17,labelpad = 5)
    axes.set_ylim(0.005,0.042)
    axes.set_xlim(-1,22)
    axes.set_yticks(np.linspace(0.00,0.04,5))
    axes.set_xticks(np.linspace(0,20,5))
    axes.set_aspect(np.diff(axes.get_xlim())/(1.5*np.diff(axes.get_ylim())))
    
    ax1.set_ylabel(r"$\text{max}\left(\vert x_{s\vert y}\vert \right) (t)$",fontsize = 17,labelpad = 5)
    ax1.set_ylim(-1e-3,1.02)
    ax1.set_xlim(-1,22)
    ax1.set_yticks(np.linspace(0.0,1,6))
    ax1.set_xticks(np.linspace(0,20,5))
    ax1.tick_params(axis = 'both',which = 'major',direction = 'in',labelsize=15,pad = 3)
    ax1.set_aspect(np.diff(ax1.get_xlim())/(1.5*np.diff(ax1.get_ylim())))
    
    for ax in [axins_0,axins_1,axins_2]:
        ax.patch.set_alpha(0)
        ax.set_xlim(-0.6,0.6)
        ax.set_ylim(-0.6,0.6)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect(np.diff(ax.get_xlim())/(np.diff(ax.get_ylim())))
    ### Shade 
    ax1.axvspan(-1,time_data[j_loop_start],color = "#EDEEC9",alpha = 0.3)
    ax1.axvspan(time_data[j_loop_start],time_data[j_loop_end],color = "#93827F",alpha = 0.3)
    ax1.axvspan(time_data[j_loop_end],22,color = "#558B6E",alpha = 0.3)
    
   
    ### Label transition points ###
    axes.text(x = 1.50,y = 0.042,s = r"\textbf{I}",size = 18)
    axes.text(x = 6.2,y = 0.042,s = r"\textbf{II}",size = 18)
    axes.text(x = 14.5,y = 0.042,s = r"\textbf{III}",size = 18)
    
    # fig.suptitle("Poiseuille-Up Flip",size = 17)
    
    pdf_filename = os.path.join(output_dir,'{}.pdf'.format(file_name))
    svg_filename = os.path.join(output_dir,'{}.svg'.format(file_name))
    

    plt.savefig(pdf_filename,
                format = 'pdf',bbox_inches = 'tight',dpi = 400)
    fig.savefig(svg_filename,
                bbox_inches = 'tight',format = 'svg',dpi = 400)
    
    plt.show()
    