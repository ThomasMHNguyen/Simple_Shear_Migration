# -*- coding: utf-8 -*-
"""
FILE NAME:                  net_ycom_comp_mu_bar_scaling.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):                    Compare_Poiseuille_Shear_Data.py


DESCRIPTION:                This script contains a function that plots the filament absolute net
                            center of mass displacement as a function of the effective shear mu_bar.
                            Poiseuille flow and shear flow data are required. For Poiseuille flow,
                            the data corresponds to 1 particular value for U_c and 1 particular value 
                            for H. 

INPUT
FILES(S):                   N/A

OUTPUT
FILES(S):       
1)                          .PNG/.PDF/.EPS file that shows net com displacement as a function
                            of the effective shear flow mu_bar. 

INPUT
ARGUMENT(S):                N/A



CREATED:                    26Apr23

MODIFICATIONS
LOG:                        
25Jun23                     Modified rescaling of axes of inset figure.
    
            
LAST MODIFIED
BY:                         Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:                     3.9.13

VERSION:                    1.1

AUTHOR(S):                  Thomas Nguyen

STATUS:                     Working

TO DO LIST:                 N/A

NOTE(S):                    N/A

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib.patches  as mpatches

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
    'text.latex.preamble': r'\usepackage{amsmath}'})


def net_ycom_comp_mu_bar_scaling(data_df,fit_slope,fit_intercept,output_dir):
    """
    This function plots the Delta y (absolute value of net displacement of center of mass)
    as a function of effective shear flow mu_bar. Curves are color coded based on
    the Flow Type (Shear or Weak/Medium/Strong Poiseuille) and Displacement Type
    (Up Vs. Down). The inset image is (Delta y)*(mu_bar_poi)^(1/8) vs. 
    (mu_bar_shear_eff/mu_bar_poi).
    
    Inputs:
        
    data_df:            Pandas dataframe that contains all of the net center
                        of mass displacement for the upward/downward flips in 
                        Poiseuille and Shear Flow Data.
    fit_slope:          Value of the slope of the best line of fit through the 
                        shear flow data.
    fit_intercept:      Value of the intercept of the best line of fit through the 
                        shear flow data.
    output_dir:         The directory where the plot will be saved to.
    
    """
    ### Filter Data ###
    fil_df = data_df[(data_df['Normalized Time'] == 1) &\
                                      (data_df['Starting Vertical Displacement'] >= 0.25)].copy()
    shear_up_df = fil_df[(fil_df['Flow Type'] == 'Shear') &\
                                (fil_df['Displacement Type'] == 'Up')].copy()
    shear_down_df = fil_df[(fil_df['Flow Type'] == 'Shear') &\
                                (fil_df['Displacement Type'] == 'Down')].copy()
    poi_up_df = fil_df[(fil_df['Flow Type'] != 'Shear') &\
                       (fil_df['Displacement Type'] == 'Up')].copy()
    poi_down_df = fil_df[(fil_df['Flow Type'] != 'Shear') &\
                       (fil_df['Displacement Type'] == 'Down')].copy()
        
    """
    Order for colors & markers are as follows:
        
    1) Poiseuille (Weak)
    2) Poiseuille (Medium)
    3) Poiseuille (Strong)
    4) Poiseuille (Shear)
    """
    
    ### Specify order of colors & markers ###
    poi_colors = ["#42BFDD",
              "#605770",
              "#ff6c5f"]
    shear_colors = ["#ffc168"]
    poi_order = ["Poiseuille (W)","Poiseuille (M)","Poiseuille (S)"]
    # shear_order = ["Shear"]
    poi_markers = ['d','v','p']
    shear_markers = ['o']
    
    fig,axes = plt.subplots(figsize = (7,7))
    axins_1 = inset_axes(axes, width="40%", height="40%", loc='upper right')
    
    ### Main plot ###
    sns.scatterplot(data = shear_down_df,x = 'Effective Shear Mu_bar',style = 'Displacement Type',
                y= 'ABS Net Adjusted Center of Mass-y',ax = axes, marker = shear_markers[0],
                facecolors = 'none',edgecolor = shear_colors[0],s = 100,
                legend = False)
    sns.scatterplot(data = shear_up_df,x = 'Effective Shear Mu_bar',style = 'Displacement Type',
                y= 'ABS Net Adjusted Center of Mass-y',ax = axes, marker = shear_markers[0],
                color = shear_colors,s = 100,legend = False)
    
    sns.scatterplot(data = poi_up_df,x = 'Effective Shear Mu_bar',style = 'Flow Type',hue = 'Flow Type',
                y= 'ABS Net Adjusted Center of Mass-y',ax = axes, markers = poi_markers,
                hue_order = poi_order,style_order = poi_order,palette = poi_colors,
                s = 100,legend = False)
    
    ### Draw line of best fit through shear flow data ###
    x = np.linspace(0.98e5,7.5e5,100)
    y = fit_intercept*x**(fit_slope)
    axes.plot(x,y,color = 'black',linestyle = 'dashed',linewidth = 1.1)
    ### Plot downward flips one at a time
    for i,v in enumerate(poi_order):
        fil_poi_down_df = poi_down_df[poi_down_df['Flow Type'] == v]
        sns.scatterplot(data = fil_poi_down_df,x = 'Effective Shear Mu_bar',
                    y= 'ABS Net Adjusted Center of Mass-y',ax = axes, marker = poi_markers[i],
                    facecolors = 'none',edgecolor = poi_colors[i],s = 100,
                    linewidth = 1.5,legend = False)
    
    ### Inset Plot ###
    sns.scatterplot(data = poi_up_df,x = 'Starting Vertical Displacement',style = 'Flow Type',hue = 'Flow Type',
                y= 'Rescaled ABS Net Adjusted Center of Mass-y',ax = axins_1, markers = poi_markers,
                hue_order = poi_order,style_order = poi_order,palette = poi_colors,
                s = 100,legend = False)
    for i,v in enumerate(poi_order):
        fil_poi_down_df = poi_down_df[poi_down_df['Flow Type'] == v]
        sns.scatterplot(data = fil_poi_down_df,x = 'Starting Vertical Displacement',
                    y= 'Rescaled ABS Net Adjusted Center of Mass-y',ax = axins_1, marker = poi_markers[i],
                    facecolors = 'none',edgecolor = poi_colors[i],s = 100,
                    linewidth = 1.5,legend = False)
    
    ### Format main plot ###
    axes.set_ylim(0.030,0.050)
    axes.set_xlim(9e4,8e5)
    axes.set(xscale = 'log',yscale = 'log')
    axes.set_ylabel(r"$\lvert \Delta y^{\text{com}}\rvert$",fontsize = 15,labelpad = 7)
    axes.set_xlabel(r"$\bar{\mu}^{\text{shear}}_{\text{eff}}$",fontsize = 15,labelpad = 7)
    axes.tick_params(axis = 'both',which = 'major',direction = 'in',labelsize = 14,pad = 5)
    axes.tick_params(axis = 'both',which = 'minor',direction = 'in',labelsize = 12,pad = 5)
    axes.set_aspect((np.diff(np.log10(axes.get_xlim())))/(np.diff(np.log10(axes.get_ylim()))))
    
    
    # # ### Format inset plot ###
    axins_1.set_ylim(1.55e-1,1.95e-1)
    axins_1.set_yticks(np.linspace(1.6e-1,1.9e-1,4))
    axins_1.set_xlim(0.22,0.48)
    axins_1.set_xticks(np.linspace(0.25,0.45,5))
    axins_1.tick_params(axis = 'both',which = 'both',direction = 'in',labelsize = 11,pad = 5)
    axins_1.set_ylabel(r"$\vert \Delta y^{\text{com}}\vert \left(\bar{\mu}^{\text{shear}}_{\text{eff}}\right)^{1/8}$",fontsize = 13,labelpad = 7)
    [l.set_visible(False) for (i,l) in enumerate(axins_1.xaxis.get_ticklabels()) if (i) % 2 != 0]
    axins_1.set_xlabel(r"$y_{0}$",fontsize = 13,labelpad = 7)
    axins_1.set_aspect((np.diff(axins_1.get_xlim()))/(np.diff(axins_1.get_ylim())))
    
    ## Custom legend ##
    all_labels = [r"$\bar{\mu}^{\text{Poi}} = 5 \times 10^{4}$",
                  r"$\bar{\mu}^{\text{Poi}} = 10^{5}$",
                  r"$\bar{\mu}^{\text{Poi}} = 2 \times 10^{5}$",
                  r"Shear Flow"]
    all_colors = []
    all_colors.extend(poi_colors)
    all_colors.extend(shear_colors)
    all_shapes = []
    all_shapes.extend(poi_markers)
    all_shapes.extend(shear_markers)
    handles = [Line2D([0], [0], color=all_colors[i], label = v,
                    linewidth = 3,linestyle = '-') for i,v in enumerate(all_labels)]
    up_flips = Line2D([0],[0],markeredgecolor = 'black',markerfacecolor = 'black',
                              linestyle = '',marker = 'o',markersize = 8,
                              label = 'Up Flips')
    down_flips = Line2D([0],[0],markeredgecolor = 'black',markerfacecolor = 'white',
                              linestyle = '',marker = 'o',markersize = 8,
                              label = 'Down Flips')
    flips_hdl = [up_flips,down_flips]
    handles.extend(flips_hdl)

    axes.legend(handles = handles,loc = 'lower left',fontsize = 12)
    plt.savefig(os.path.join(output_dir,'net_com_all_APS.png'),
                format = 'png',dpi = 400,bbox_inches = 'tight')
    plt.savefig(os.path.join(output_dir,'net_com_all_APS.pdf'),
                format = 'pdf',dpi = 400,bbox_inches = 'tight')
    plt.savefig(os.path.join(output_dir,'net_com_all_APS.eps'),
                dpi = 400,bbox_inches = 'tight',format = 'eps')
    plt.show()
