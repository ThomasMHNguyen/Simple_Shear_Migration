# -*- coding: utf-8 -*-
"""
FILE NAME:              radius_mu_bar_scaling_subplots.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):                A__v01_03_Process_Filament_Flip_Data.py

DESCRIPTION:            This script contains a function that will plot the radius of 
                        bending of the J-shapes as a function of the adjusted mu_bar value
                        (mu_bar times center of mass prior to rotation). Inset images will
                        show the scaling relationship between radius of bending and regular
                        mu_bar.

INPUT
FILES(S):               N/A

OUTPUT
FILES(S):               1) .PNG/.PDF/.EPS file that shows the scatterplot relationship between
                        radius of bending and adjusted mu_bar.
                

INPUT
ARGUMENT(S):            N/A

CREATED:                22Nov22

MODIFICATIONS
LOG:
22Nov22                 1) Migrated code to generate the plots from the original script
                        to its own instance here.

    
            
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

### Use LaTeX to generate plots ###
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})


#%%
def radius_mu_bar_scaling_subplots(input_file,fit_params,output_directory,file_name):
    """
    This function will take in the filament flipping data and plot the radius 
    of bending as a function of adjusted mu_bar.
    
    Inputs:
    
    input_file:                 Input Pandas dataframe that lists the measured
                                data values for each U-turn flipping event.
    fit_params:                 Python dictionary that contains the slope and intercept
                                of the fitted data (Shear/Poiseuille flow radius to regular/
                                spatial mu_bar)
    output_directory:           Path to directory to be created.
    file_name:                  The file name to be used for the output graphical
                                files.
    """
    
    shear_df = input_file[input_file['Flow Type'] == 'Shear']
    poiseuille_df = input_file[input_file['Flow Type'] == 'Poiseuille']
    
    ### Custom ordering shapes & colors
    hue_order = [5e4,1e5,2e5,5e5]
    hue_order_fix_all = [r'$5 \times 10^{4}$',r'$1 \times 10^{5}$',
                 r'$2 \times 10^{5}$',r'$5 \times 10^{5}$']

    poi_colors = ["#57886C","#F2545B","#713E5A","#78C0E0"]
    shear_colors = poi_colors.copy()
    
    poi_markers = ['d','v','p','o']
    shear_markers = poi_markers.copy()
    

    ### Set up Canvas space ###
    fig,axes = plt.subplots(ncols = 2,figsize = (10,7),sharey = True,layout = 'constrained')
    axins_0 = inset_axes(axes[0], width="45%", height="45%", loc='upper right')
    axins_1 = inset_axes(axes[1], width="45%", height="45%", loc='upper right')
    
    
    ### main plot ###
    sns.scatterplot(x = 'Mu_bar times Starting COM-y',y = 'Radius of Bending-1',
                               data = poiseuille_df,hue = 'Mu_bar String',
                               hue_order = hue_order_fix_all,style = 'Mu_bar String',
                               markers = poi_markers,palette = poi_colors,
                               s = 80,alpha = 0.85,
                               ax = axes[0])
    sns.scatterplot(x = 'Mu_bar times Starting COM-y',y = 'Radius of Bending-1',
                               data = shear_df,hue = 'Mu_bar String',
                               hue_order = hue_order_fix_all,style = 'Mu_bar String',
                               markers = shear_markers,palette = shear_colors,
                               s = 80,alpha = 0.85,
                               ax = axes[1],legend = False)
    ###  Insets ###
    for i,v in enumerate(hue_order):
        fil_poi_df = poiseuille_df[poiseuille_df['Mu_bar'] == v]
        fil_shear_df = shear_df[shear_df['Mu_bar'] == v]
        sns.lineplot(x = 'Mu_bar',y = 'Radius of Bending-1',
                                    data = fil_poi_df,color = poi_colors[i],
                                    marker = poi_markers[i],markersize = 8,
                                    err_style="bars",
                                    linestyle = '',legend = False,
                                    err_kws={'capsize':5,'capthick': 2},
                                    errorbar=("sd", 1),ax = axins_0)
        sns.lineplot(x = 'Mu_bar',y = 'Radius of Bending-1',
                                    data = fil_shear_df,color = shear_colors[i],
                                    marker = shear_markers[i],markersize = 8,
                                    err_style="bars",
                                    linestyle = '',legend = False,
                                    err_kws={'capsize':5,'capthick': 2},
                                    errorbar=("sd", 1),ax = axins_1)
        
    ### Format main axes ###
    for i,ax in enumerate(axes):
        x = np.linspace(1.3e2,1e6,100)
        if i == 0:
            y = fit_params["Poiseuille Spatial Intercept"]*x**fit_params['Poiseuille Spatial Slope']
        elif i == 1:
            y = fit_params["Shear Spatial Intercept"]*x**fit_params['Shear Spatial Slope']
        ax.plot(x,y,color = 'black',linestyle = 'dashed',
                  linewidth = 1.2)
        ax.set_ylim(3e-2,2e-1)
        ax.set_xlim(1e3,1.2e6)
        ax.set(xscale = 'log',yscale = 'log')
        ax.tick_params(axis='both', which='major', direction = 'in',labelsize=13,pad = 5)
        ax.tick_params(axis='both', which='minor', direction = 'in',labelsize=11,pad = 5)
    
        ax.set_aspect((np.diff(np.log(ax.get_xlim())))/(np.diff(np.log(ax.get_ylim()))))
        ax.set_xlabel(r"$\bar{\mu}y_{i}^{\text{com}}$",fontsize = 17,labelpad = 5)
        ax.set_ylabel(r"$R$",fontsize = 17,labelpad = 5)
        # ax.text(x = 1.42e3,y = 1.2e-1,s = r'$\propto \left(\bar{\mu}y^{\text{com}}_{i}\right)^{-1/4}$',size = 15)
        if i == 0:
            legend = ax.legend(loc='lower left', 
                    prop={'size': 15},title= r"$\bar{\mu}$")
            legend.get_title().set_fontsize("17")
            # for handle in legend.legendHandles:
            #     handle.set_edgecolor("black")
    
    ### Format inset axes ###
    for i,ax in enumerate([axins_0,axins_1]):
        x = np.linspace(1e4,1.2e6,100)
        if i == 0:
            y1 = fit_params["Poiseuille Regular Intercept"]*x**fit_params['Poiseuille Regular Slope']
        elif i == 1:
            y1 = fit_params["Shear Regular Intercept"]*x**fit_params['Shear Regular Slope']
        ax.set_xlim(5e3,1.5e6)
        ax.set_ylim(3.2e-2,1.005e-1)
        ax.plot(x,y1,color = 'black',linewidth = 0.9,linestyle = 'dashed')
        ax.set(xscale = 'log',yscale = 'log')
        ax.set_xlabel(r"$\bar{\mu}$",fontsize = 13,labelpad = 5)
        ax.set_ylabel(r"$R$",fontsize = 13,labelpad = 5)
        ax.tick_params(axis='both', which='major', direction = 'in',labelsize=11,pad = 3)
        ax.tick_params(axis='both', which='minor', direction = 'in',labelsize=9,pad = 3)
        # ax.text(x = 1.2e4,y = 3.75e-2,s = r'$\propto \left(\bar{\mu}\right)^{-1/4}$',size = 12)
    
    ### Labels for subplots ###
    axes[0].text(x = 1.70e2,y = 1.9e-1,s = r'\textbf{(a)}',size = 15)
    axes[1].text(x = 6.30e2,y = 1.9e-1,s = r'\textbf{(b)}',size = 15)
    fig.savefig(os.path.join(output_directory,'{}.png'.format(file_name)),
                bbox_inches = 'tight',dpi = 600)
    fig.savefig(os.path.join(output_directory,'{}.pdf'.format(file_name)),
                format = 'pdf',bbox_inches = 'tight',dpi = 600)
    fig.savefig(os.path.join(output_directory,'{}.eps'.format(file_name)),
                bbox_inches = 'tight',format = 'eps',dpi = 600)
    plt.show()