# -*- coding: utf-8 -*-
"""
FILE NAME:      plot_com_time_curves.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        Compare_Poiseuille_Shear_COM.py

DESCRIPTION:    In Poiseuille flow, for all simulations at the same mu_bar value, 
                this function will plot the center of mass time curves 
                between the upward and downward flips. A color bar is generated 
                to denote the initial filament starting position.

INPUT
FILES(S):       N/A

OUTPUT
FILES(S):       1) .PNG/.PDF/.EPS file that shows center of mass time curves for 
                Poiseuille flow at various starting vertical displacements.

INPUT
ARGUMENT(S):    N/A


CREATED:        26Apr23

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

NOTE(S):        N/A

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
    'text.latex.preamble': r'\usepackage{amsmath}'})


def ycom_time_curves_poi_all(data_df,output_dir,poi_flow_type,
                         poi_mu_bar,poi_Uc):
    """
    In Poiseuille flow, for all simulations at the same mu_bar value, this function
    will plot the center of mass time curves between the upward and downward
    flips. A color bar is generated to denote the initial filament starting 
    position.
    
    Inputs:
        
    position_vals:          1x5 Numpy array that contains starting vertical 
                            displacement values to plot the center of mass
                            values for.
    """
    # Filter for specific data
    fil_df = data_df[(data_df['Flow Type'] == 'Poiseuille (M-C)') &\
                              (data_df['Mu_bar'] == 1e5) &\
                                  (data_df['Poiseuille U Centerline'] == 1)]
    fil_df = fil_df[(fil_df['Time'] <= 20) &\
                    (fil_df['Time'].isin(np.arange(0,20.01,0.10)))]    
    
    #Set up colors for plot and color for color bar
    palette_t = 'inferno_r'
    cmap_colors = sns.color_palette(palette_t,
                                    n_colors = fil_df['Starting Vertical Displacement'].unique().shape[0])
    
    fig,axes = plt.subplots(figsize = (10,7),ncols = 2,sharey = True,layout = 'constrained')
    
    ### Plot Data ###
    sns.lineplot(data = fil_df[fil_df['Displacement Type'] == 'Up'],x = 'Time',
                 y = 'ABS Net Adjusted Center of Mass-y',
                  hue = 'Starting Vertical Displacement',palette = cmap_colors,
                  ax = axes[0],legend = False,linewidth = 2.5)
    rhs_plot = sns.lineplot(data = fil_df[fil_df['Displacement Type'] == 'Down'],
                            x = 'Time',y = 'ABS Net Adjusted Center of Mass-y',
                  hue = 'Starting Vertical Displacement',palette = cmap_colors,
                  ax = axes[1],linewidth = 2.5)
    rhs_plot.legend().remove()
    
    ### Color bar ###
    norm = plt.Normalize(fil_df['Starting Vertical Displacement'].min(),
                         fil_df['Starting Vertical Displacement'].max())
    sm = plt.cm.ScalarMappable(cmap=palette_t, norm=norm)
    sm.set_array([])

    subplot_texts = [r"\textbf{(a)}",r"\textbf{(b)}"]
    
    ### Format axes ###
    for i,ax in enumerate(axes):
        ax.set_ylim(0.005,0.042)
        ax.set_xlim(-2,22)
        ax.set_yticks(np.linspace(0.01,0.04,4))
        ax.set_xticks(np.linspace(0,20,5))
        ax.tick_params(axis = 'both',which = 'both',direction = 'in',
                       labelsize = 19,pad = 5,width = 1.2,length = 5)
        ax.set_aspect((np.diff(ax.get_xlim()))/(1.25*np.diff(ax.get_ylim())))
        ax.set_xlabel("")
        ax.set_xlabel(r"$t$",size = 21)
        ax.text(x = -1.5,y = 0.043,s = subplot_texts[i],size = 21)
        
        axes[0].set_ylabel(r"$\lvert y^{\text{com}} \: (t) - y_{0} \rvert $", x = 0.05,size = 21)  
    
    ### Format color bar ###
    cbar_ax = fig.add_axes([0.08, 0.08, 0.9, 0.05])  # Adjust the position and size as needed (x0,y0,width,height)
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=np.arange(0.25, 0.46, 0.05), orientation='horizontal')
    cbar.set_label(r"$y_{0}$", fontsize=21)
    cbar.ax.tick_params(labelsize=19)  # Specify the desired font size
    
    # axes[0].set_ylabel(r"$\vert y^{\text{com}} \: (t) - y_{0} \vert $", x = 0.05,size = 16)  
    # fig.supxlabel(r"$t_{\text{sim}}$",y = 0.20,x = 0.45,size = 16)
    # fig.supxlabel(r"$t$",y = 0.18,x = 0.48,size = 16)
    # fig.suptitle(r"$\bar{{\mu}}^{{\text{{Poiseuille}}}} = {0:.2e}$" "\n" r"$U_{{x}} = {1:.2f}\left(1-y^{{2}}/{2:.2f}^{{2}}\right)$".format(
    #                 poi_mu_bar,poi_Uc,fil_df['Channel Height'].unique()[0]),size = 13,y = 0.98)
    plt.savefig(os.path.join(output_dir,'net_com_comp.png'),
                dpi = 600,bbox_inches = 'tight')
    plt.savefig(os.path.join(output_dir,'net_com_comp.pdf'),
                format = 'pdf',dpi = 600,bbox_inches = 'tight')
    plt.savefig(os.path.join(output_dir,'net_com_comp.eps'),
                dpi = 600,bbox_inches = 'tight',format = 'eps')
        
    plt.show()
        
