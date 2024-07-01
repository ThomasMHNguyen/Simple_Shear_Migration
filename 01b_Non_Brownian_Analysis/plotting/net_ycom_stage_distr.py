# -*- coding: utf-8 -*-
"""
FILE NAME:              net_ycom_stage_distr.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):                Compare_Poiseuille_Shear_Data.py

DESCRIPTION:            This script contains functions to plot the net center
                        of mass displacement of the filament in Poiseuille flow at
                        at particular flow strength to show the differences in drift
                        between the stages. The displacement will be plotted on a 
                        strip plot (categorial scatter plot).

INPUT
FILES(S):               N/A

OUTPUT
FILES(S):       
    
1)                      .PNG/.PDF/.EPS of a strip/swarm plot that shows the net center of
                        mass displacement curves based on the stage of the U-turn as well
                        as whether or not the filament performed an upward or downward flip. 

INPUT
ARGUMENT(S):            N/A
                

CREATED:                20Jun23

MODIFICATIONS
LOG:                    

18Jul23                 1) Added new function to plot filament shape snapshots as subplot
                        on Net drift violinplot. Swarmplots can now replace the violin
                        plots. 
    
            
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

NOTE(S):        
    
1)                      Running the method to annotate the plots for statistical significance requires
                        the library 'statsannotations'. The current version of 'statsannotations'
                        library (v0.5) requires 'seaborn' (<v0.12). If running this script
                        through conda, a separate environment is required.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

### Use LaTeX to generate plots ###
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
    'text.latex.preamble': r'\usepackage{amsmath}'})


def net_ycom_stage_distr(input_file,p_vals_df,output_directory,file_name):
    """
    This function will take in the filament flipping data and plot the net 
    displacement as as function of mu_bar on a stripplot (categorical scatter
    plot).
    
    Inputs:
    
    input_file:                 Pandas dataframe that lists the measured
                                data values for each U-turn flipping event.
    p_vals_df:                  Pandas DataFrame that lists the FDR-adjusted
                                p-values for each valid comparison.
    output_directory:           Path to directory to be created.
    file_name:                  The file name to be used for the output graphical
                                files.
    """

    hue_order_fix = ["I","II","III"]
    cmap_palette = ["#79ADDC","#F3DFA2"]
    p_vals_df.sort_values(by='Stage_1', key=lambda x: x.map({val: i for \
                                                             i, val in enumerate(hue_order_fix)}),
                          inplace = True)
    
    fig,axes = plt.subplots(figsize = (10,7))

        
    #Plot Swarmplot with Line to show mean data
    g = sns.stripplot(y = 'Stage',x = 'Stage ABS Adjusted Center of Mass-y',
                                data = input_file,hue = 'Displacement Type',
                                palette = cmap_palette,dodge = True,jitter = 0.25,
                                order = hue_order_fix,size = 9,edgecolor = 'gray',
                                alpha = 0.8,linewidth = 1.2,ax = axes)
    g.get_legend().remove()
    mean_props = dict(linestyle='solid', linewidth=2.5, color='black')
    g = sns.boxplot(showmeans=True,
            meanline=True,
            meanprops=mean_props,
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            y="Stage",
            x="Stage ABS Adjusted Center of Mass-y",
            data=input_file,
            hue = 'Displacement Type',
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=axes)
    
    # dummy plots, just to get the Path objects
    a = axes.scatter([0,1],[0,1], marker='o')
    b = axes.scatter([0,1],[0,1], marker='d')
    circle_mk, = a.get_paths()
    diamond_up_mk, = b.get_paths()
    a.remove()
    b.remove()
    c = axes.collections
    for i,v in enumerate(c):
        if i % 2 == 0:
            v.set_paths([circle_mk])
        if i % 2 != 0:
            v.set_paths([diamond_up_mk])
            
    #Custom legend to reflect shapes
    up_handle = Line2D(xdata = [0],ydata = [0],markerfacecolor=cmap_palette[0],
                       label='Up',marker = 'o',markeredgecolor = 'gray',
                       markersize = 9,linestyle = '')
    down_handle = Line2D(xdata = [0],ydata = [0],markerfacecolor=cmap_palette[1],
                         label='Down',marker = 'd',markeredgecolor = 'gray',
                         markersize = 9,linestyle = '')
    axes.legend(handles = [up_handle,down_handle],loc='upper right', 
                prop={'size': 15},title= "Flip Direction").get_title().set_fontsize("16")
    
    
    
    #General axes formatting    
    axes.tick_params(axis='both', which='both', direction = 'in',labelsize=15,pad = 5)
    axes.ticklabel_format(axis="x", style="sci", scilimits=(-3,-3))
    axes.xaxis.offsetText.set_visible(False)
    axes.set_xlim(5.8e-3,1.51e-2)
    axes.set_xticks(np.linspace(6e-3,1.5e-2,4))
    axes.set_ylabel(r"Movement Phase",fontsize = 17,labelpad = 5)
    axes.set_xlabel(r"$|\Delta y^{\text{com}}| \times 10^{-3}$",fontsize = 17,labelpad = 5)
    axes.set_aspect((np.diff(axes.get_xlim()))/(1.1*np.abs(np.diff(axes.get_ylim()))))
     
    ### Draw Lines to Show Significant p-values ###
     
    pairs = [((i[-4],i[-3]),(i[-2],i[-1])) for i in p_vals_df.itertuples()]
    pvals = [i[4] for i in p_vals_df.itertuples()]
    annotator = Annotator(
    axes, pairs, data=input_file, y = 'Stage',x = 'Stage ABS Adjusted Center of Mass-y',
    hue = 'Displacement Type',orient = 'h',order = hue_order_fix)
    annotator.configure(text_format="star", loc="outside",fontsize = 'x-large',line_height = 0)
    annotator.set_pvalues(pvals)
    annotator.annotate()
            
    fig.savefig(os.path.join(output_directory,'{}.png'.format(file_name)),
                bbox_inches = 'tight',dpi = 400)
    fig.savefig(os.path.join(output_directory,'{}.pdf'.format(file_name)),
                format = 'pdf',bbox_inches = 'tight',dpi = 400)
    fig.savefig(os.path.join(output_directory,'{}.eps'.format(file_name)),
                bbox_inches = 'tight',format = 'eps',dpi = 400)
    plt.show()


def net_ycom_stage_distr_vert(input_file,p_vals_df,output_directory,file_name):
    """
    This function will take in the filament flipping data and plot the net 
    displacement as as function of mu_bar on a stripplot (categorical scatter
    plot). The stripplot will be oriented vertically.
    
    Inputs:
    
    input_file:                 Pandas dataframe that lists the measured
                                data values for each U-turn flipping event.
    p_vals_df:                  Pandas DataFrame that lists the FDR-adjusted
                                p-values for each valid comparison.
    output_directory:           Path to directory to be created.
    file_name:                  The file name to be used for the output graphical
                                files.
    """
    hue_order_fix = ["I","II","III"]
    cmap_palette = ["#79ADDC","#F3DFA2"]
    p_vals_df.sort_values(by='Stage_1', key=lambda x: x.map({val: i for i, val in enumerate(hue_order_fix)}),
                          inplace = True)
    
    fig,axes = plt.subplots(figsize = (7,10))        
    #Plot Swarmplot with Line to show mean data
    g = sns.stripplot(x = 'Stage',y = 'Stage ABS Adjusted Center of Mass-y',
                                data = input_file,hue = 'Displacement Type',
                                palette = cmap_palette,dodge = True,jitter = 0.25,
                                order = hue_order_fix,size = 9,edgecolor = 'gray',
                                alpha = 0.8,linewidth = 1.2,ax = axes)
    g.get_legend().remove()

    mean_props = dict(linestyle='solid', linewidth=2.5, color='black')
    g = sns.boxplot(showmeans=True,
            meanline=True,
            meanprops=mean_props,
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="Stage",
            y="Stage ABS Adjusted Center of Mass-y",
            data=input_file,
            hue = 'Displacement Type',
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=axes)
    
    # dummy plots, just to get the Path objects
    a = axes.scatter([1,2],[3,4], marker='o')
    b = axes.scatter([1,2],[3,4], marker='d')
    circle_mk, = a.get_paths()
    diamond_up_mk, = b.get_paths()
    a.remove()
    b.remove()
    c = axes.collections
    for i,v in enumerate(c):
        if i % 2 == 0:
            v.set_paths([circle_mk])
        if i % 2 != 0:
            v.set_paths([diamond_up_mk])
            
    #Custom legend to reflect shapes
    up_handle = Line2D(xdata = [0],ydata = [0],markerfacecolor=cmap_palette[0],
                       label='Up',marker = 'o',markeredgecolor = 'gray',
                       markersize = 9,linestyle = '')
    down_handle = Line2D(xdata = [0],ydata = [0],markerfacecolor=cmap_palette[1],
                         label='Down',marker = 'd',markeredgecolor = 'gray',
                         markersize = 9,linestyle = '')
    axes.legend(handles = [up_handle,down_handle],loc='upper right', 
                prop={'size': 15},title= "Flip Direction").get_title().set_fontsize("16")
    
    #General figure formatting
    axes.tick_params(axis='both', which='both', direction = 'in',labelsize=15,pad = 5)
    axes.ticklabel_format(axis="y", style="sci", scilimits=(-3,-3))
    axes.yaxis.offsetText.set_visible(False)
    axes.set_ylim(5.8e-3,1.51e-2)
    axes.set_yticks(np.linspace(6e-3,1.5e-2,4))
    axes.set_xlabel(r"Movement Phase",fontsize = 17,labelpad = 5)
    axes.set_ylabel(r"$|\Delta y^{\text{com}}| \times 10^{-3}$",fontsize = 17,labelpad = 5)
    
    axes.set_aspect(1.1*(np.diff(axes.get_xlim()))/(np.diff(axes.get_ylim())))
     
    ### Draw Lines to Show Significant p-values ###
     
    pairs = [((i[-4],i[-3]),(i[-2],i[-1])) for i in p_vals_df.itertuples()]
    pvals = [i[4] for i in p_vals_df.itertuples()]
    annotator = Annotator(
    axes, pairs, data=input_file, x = 'Stage',y = 'Stage ABS Adjusted Center of Mass-y',
    hue = 'Displacement Type',order = hue_order_fix)
    annotator.configure(text_format="star", loc="outside",fontsize = 'x-large',line_height = 0)
    annotator.set_pvalues(pvals)
    annotator.annotate()
            
    fig.savefig(os.path.join(output_directory,'{}.png'.format(file_name)),
                bbox_inches = 'tight',dpi = 400)
    fig.savefig(os.path.join(output_directory,'{}.pdf'.format(file_name)),
                format = 'pdf',bbox_inches = 'tight',dpi = 400)
    fig.savefig(os.path.join(output_directory,'{}.eps'.format(file_name)),
                bbox_inches = 'tight',format = 'eps',dpi = 400)
    plt.show()
