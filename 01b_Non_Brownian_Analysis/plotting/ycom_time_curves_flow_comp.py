# -*- coding: utf-8 -*-
"""
FILE NAME:      plot_com_time_curves.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        Compare_Poiseuille_Shear_Data.py

DESCRIPTION:    At the same mu_bar value, this function plots the filament 
                center of mass in shear and Poiseuille flow performing up and 
                down flips as a function of time on subplots. As is it currently 
                set up, this function will plot only if 5 vertical displacement 
                values are supplied.

INPUT
FILES(S):       N/A

OUTPUT
FILES(S):       1) .PNG/.PDF/.EPS file that shows center of mass curves for 
                Poiseuille for 5 equivalent starting vertical displacements 
                and the equivalent mu_bar in shear flow.

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
    "font.sans-serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})


def ycom_time_curves_flow_comp(data_df,output_dir,position_vals,poi_flow_type,
                         poi_mu_bar,poi_Uc):
    """
    At the same mu_bar value, this function plots the filament center of mass 
    in shear and Poiseuille flow performing up and down flips as a function 
    of time on subplots. As is it currently set up, this function will plot only
    if 5 vertical displacement values are supplied.
    
    Inputs:
        
    data_df:                DataFrame that contains both the simulation time
                            and center of mass data for both shear and 
                            Poiseuille flow.
    output_dir:             Output directory for the files to be saved in.
    position_vals:          Array of values that represent the initial filament
                            starting position to plot the data for.
    poi_flow_type:          String that represents Poiseuille flow strength to
                            plot the data for.
    poi_mu_bar:             The flow strength of Poiseuille flow to filter the
                            data for.
    poi_Uc:                 The centerline velocity of Poiseuille flow to filter
                            the data for.
    """
    if position_vals.shape[0] == 5:
        fig,axes = plt.subplots(figsize = (10,7),nrows = 2,ncols = 3,
                                sharey = True,sharex = True)
        fil_df = data_df[(data_df['Flow Type'] == 'Shear') |\
                                 ((data_df['Flow Type'] == poi_flow_type) &\
                                  (data_df['Mu_bar'] == poi_mu_bar) &\
                                      (data_df['Poiseuille U Centerline'] == poi_Uc))]
        fil_df = fil_df[(fil_df['Brownian Time'] <= 40) &\
                        (fil_df['Brownian Time'].isin(np.arange(0,40.01,0.10)))]
        fil_df['Flow Type-Displacement Type'] = fil_df['Flow Type-Displacement Type'].replace("Poiseuille (M-C)-Up",'Poiseuille-Up')
        fil_df['Flow Type-Displacement Type'] = fil_df['Flow Type-Displacement Type'].replace("Poiseuille (M-C)-Down",'Poiseuille-Down')
        fig.subplots_adjust(hspace = -0.2,wspace = 0.25)
        n_plot_counter = -1
        for n_row,ax_row in enumerate(axes):
            for n_col,ax_col in enumerate(ax_row):
                if n_col != 2 or n_row != 1:
                    n_plot_counter += 1
                    shr_mu_bar = np.round(position_vals[n_plot_counter]*8*poi_mu_bar)
                    fil_data_df = fil_df[(
                        fil_df['Effective Shear Mu_bar'] == shr_mu_bar)] 
                    if n_col == 1 and n_row == 1:
                        sns.lineplot(data = fil_data_df,x = 'Brownian Time',
                                     y = 'ABS Net Adjusted Center of Mass-y',
                                     hue = 'Flow Type-Displacement Type',palette = 'Set1',
                                     ax = ax_col,legend = True)
                    else:
                        sns.lineplot(data = fil_data_df,x = 'Brownian Time',
                                     y = 'ABS Net Adjusted Center of Mass-y',
                                     hue = 'Flow Type-Displacement Type',palette = 'Set1',
                                     ax = ax_col,legend = False)
                    ax_col.set_ylabel("")
                    ax_col.set_xlabel("")
        mu_bar_counter = -1
        for n_row,ax_row in enumerate(axes):
            for n_col,ax_col in enumerate(ax_row):
                if n_col != 2 or n_row != 1:
                    mu_bar_counter += 1
                    shr_mu_bar = np.round(position_vals[mu_bar_counter]*8*poi_mu_bar)
                    ax_col.set_title(r'$y^{{\text{{Poiseuille}}}}_{{0}} = {0:.2f} \vert \bar{{\mu}}^{{\text{{shear}}}}_{{\text{{eff}}}} = {1:.1e}$'.format(
                                    position_vals[mu_bar_counter],shr_mu_bar),fontsize = 12)
                    ax_col.set_ylim(0.005,0.045)
                    ax_col.set_xlim(-2,42)
                    ax_col.set_yticks(np.linspace(0.01,0.04,4))
                    ax_col.set_xticks(np.linspace(0,40,5))
                    ax_col.tick_params(axis = 'both',which = 'both',direction = 'in')
                    ax_col.set_aspect((np.diff(ax_col.get_xlim()))/(1.25*np.diff(ax_col.get_ylim())))
        fig.delaxes(axes[-1,-1])
        fig.supxlabel(r"$t_{\text{sim}}$",y = 0.15,x = 0.45,size = 15)
        fig.supylabel(r"$|y^{\text{com}} - y_{0}|$", x = 0.05,size = 15)
        # fig.suptitle(r"$\bar{{\mu}}^{{\text{{Poiseuille}}}} = {0:.2e}$" "\n" r"$U_{{x}} = {1:.2f}\left(1-y^{{2}}/{2:.2f}^{{2}}\right)$".format(
        #                 poi_mu_bar,poi_Uc,fil_df['Channel Height'].unique()[0]),size = 13,y = 0.94)
        axes[1,1].legend(loc = 'lower left', bbox_to_anchor=(1, 0.25),
                    prop={'size': 13},title= "Flow-Direction",title_fontsize = 15)
        plt.savefig(os.path.join(output_dir,'shear_poi_com_time_comp.png'),
                    dpi = 400,bbox_inches = 'tight')
        plt.savefig(os.path.join(output_dir,'shear_poi_com_time_comp.pdf'),
                    dpi = 400,bbox_inches = 'tight')
        plt.savefig(os.path.join(output_dir,'shear_poi_com_time_comp.eps'),
                    dpi = 400,bbox_inches = 'tight',format = 'eps')
        plt.show()