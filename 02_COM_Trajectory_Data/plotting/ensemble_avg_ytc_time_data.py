# -*- coding: utf-8 -*-
"""
FILE NAME:              ensemble_avg_ytc_time_data.py
    

COMPLEMENTARY
SCRIPT(S)/
FILE(S):                Process_COM_Migration_Data.py


DESCRIPTION:            This script contains a function that plots the instantaneous
                        ensemble average y-true center data as a function of 
                        time; the data curve are color-coded based on the initial 
                        starting position.

INPUT
FILES(S):               N/A

OUTPUT
FILES(S):               1) .PNG/.PDF/.EPS file that shows the ensemble average 
                        y-true center trajectories.


INPUT
ARGUMENT(S):            N/A


CREATED:                22Nov22


MODIFICATIONS
LOG:
22Nov22                 Migrated code to generate the plots from the 
                        original script to its own instance here.
25Sep23                 Separate the y-COM and y-true center functions into 
                        different scripts.

            
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": "Times New Roman",
#     'text.latex.preamble': r'\usepackage{amsmath}'})
plt.rcdefaults()


def ensemble_avg_ytc_time_data(ensemble_data_df,output_dir):
    """
    This function plots the ensemble average true center trajectory of a 
    filament based on a particular rigidity profile, 
    flow strength value, channel height, starting vertical displacement, 
    steric velocity exponential coefficient, flow type, and whether or not
    the sterics algorithm was used or not.
    
    Inputs:
        
    ensemble_data_df:           DataFrame that contains all ensemble
                                average data.
    output_directory:           Directory where the generated plots will be stored in.
    """
    
    #Group by Rigidity Profile, Mu_bar,Channel Height, Vertical Displacement, Velocity Exponential Coefficient
    exp_groups = ensemble_data_df.groupby(
        by = ['Rigidity Suffix','Mu_bar','Channel Height',  #0-2
              'Poiseuille U Centerline','Kolmogorov Phase Text', #3-4
              'Kolmogorov Phase Value','Kolmogorov Frequency', #5-6
              'Steric Velocity Exponential Coefficient','Steric Velocity Gap Criteria', #7-8
              'Sterics Use','Flow Type']) #9-10
    for group in exp_groups.groups.keys():
        group_df = exp_groups.get_group(group)
        rigid,mu_bar,channel_h,u_centerline,k_phase_text,k_phase_val,k_freq = group[:7]
        steric_use,flow_type = group[-2:]

        str_mu_bar_v1 = '{:.1e}'.format(mu_bar).split('e')
        if len(str_mu_bar_v1[0].split('.')[1]) == 1:
            pref = '{0}p{1}{2}'.format(str_mu_bar_v1[0].split('.')[0],'0',
                                       str_mu_bar_v1[0].split('.')[1])
        else:
            pref = '{0}p{1}'.format(str_mu_bar_v1[0].split('.')[0],
                                       str_mu_bar_v1[0].split('.')[1])
        mu_bar_text = '{0}{1}'.format(pref,str_mu_bar_v1[1].replace('+','e'))
        U_c_text = '{:.2f}'.format(u_centerline).replace('.','p')
        channel_h_text = '{:.2f}'.format(channel_h).replace('.','p')
        ##### Plot All Flows #####
        if flow_type == 'Shear':
            abr_flow_txt = 'SHR'
            filename_prefix = '_{}_ST_{}_MB_{}_H_{}'.format(steric_use,
                                                            abr_flow_txt,
                                                            mu_bar_text,
                                                            channel_h_text)
            if steric_use == 'Yes':
                figure_title = r"$\langle y^{{\text{{com}}}} \rangle \text{{vs. t^{{Br}}}} \vert \bar{{\mu}} = {0:.1e}$" "\n" r"$H = {1} \vert U_{{x}} = y$".format(
                mu_bar,channel_h)
            else:
                figure_title = r"$\langle y^{{\text{{com}}}} \rangle \text{{vs. t^{{Br}}}} \vert \bar{{\mu}} = {0:.1e}$" "\n" "U_{{x}} = y".format(mu_bar)
        elif flow_type == 'Poiseuille':
            abr_flow_txt = 'POI'
            filename_prefix = '{}_ST_{}_MB_{}_H_{}_UC_{}'.format(steric_use,
                                                                 abr_flow_txt,
                                                                 mu_bar_text,
                                                                 channel_h_text,
                                                                 U_c_text)
            if steric_use == 'Yes':
                figure_title = r"$ \langle y^{{\text{{com}}}} \rangle$ vs. $ t^{{\text{{Br}}}}\: \vert \: \bar{{\mu}} = {0:.1e}$" "\n" r"$H = {1:.2f} \vert U_{{x}} = {2:.2f}(1-y^{{2}}/{3:.2f}^{{2}})$".format(
                mu_bar,channel_h,u_centerline,channel_h)
            else:
                figure_title = r"$ \langle y^{{\text{{com}}}} \rangle$ vs. $ t^{{\text{{Br}}}}\: \vert \: \bar{{\mu}} = {0:.1e}$" "\n" "U_{{x}} = {1:.2f}(1-y^{{2}}/{2:.2f}^{{2}})$".format(
                    mu_bar,u_centerline,channel_h)
        elif flow_type == 'Kolmogorov':
            abr_flow_txt = 'KMG'
            filename_prefix = '{}_ST_{}_MB_{}_H_{}_FR_{}_PH_{}'.format(steric_use,
                                                                       abr_flow_txt,
                                                                       mu_bar_text,
                                                                       channel_h_text,
                                                                       int(k_freq),
                                                                       k_phase_text)
        
            if steric_use == 'No':
                figure_title = r"$\langle y^{{\text{{com}}}} \rangle \text{{vs. t^{{Br}}}} \vert \bar{{\mu}} = {0:.1e}$" "\n" r"$H = {1:.2f} \vert U_{{x}} = \text{{sin}}\left({2} \times {3:.0f} y\right)$".format(
                mu_bar,channel_h,k_phase_text,k_freq)

        ##### Center of Mass #####
        fig,axes = plt.subplots(figsize = (7,7))
        sns.lineplot(y = 'True Center-y',x = 'Brownian Time',hue = 'Starting Vertical Displacement',
                                palette = 'tab20',data = group_df,linewidth = 2.5,
                                legend = False,ax = axes)
        axes.ticklabel_format(axis="x", style="sci", scilimits=(-2,-2))
        axes = plt.gca()

        axes.set_title(figure_title,fontsize = 15,pad = 5)
        axes.set_xlabel(r"$t^{{\text{Br}}} \times 10^{{-2}}$",fontsize = 13,labelpad = 13)
        axes.set_ylabel(r"$\langle y^{\text{com}} \rangle $",fontsize = 13,labelpad = 13)
        axes.tick_params(axis='both', which='major', labelsize=11,direction = 'in')
        axes.xaxis.offsetText.set_fontsize(0)
        
        ### Set limits for time display ###
        if group_df['Brownian Time'].max() == 5e-2:
            axes.set_xlim(-1e-6,5.05e-2)
            axes.set_xticks(np.linspace(0,5e-2,6))
        elif group_df['Brownian Time'].max() == 1e-1:
            axes.set_xlim(-5e-6,1.05e-1)
            axes.set_xticks(np.linspace(0,1e-1,6))
        elif group_df['Brownian Time'].max() == 1.5e-1:
            axes.set_xlim(-1e-5,1.7e-1)
            axes.set_xticks(np.linspace(0,1.5e-1,4))
        elif group_df['Brownian Time'].max() == 2e-1:
            axes.set_xlim(-5e-6,2.2e-1)
            axes.set_xticks(np.linspace(0,2e-1,5))
        
        ### Set limits for channel height ###
        if channel_h == 0.25 and flow_type != 'Kolmogorov':
            axes.set_ylim(-0.3,0.3)
            axes.set_yticks(np.linspace(-0.25,0.25,3))     
        if channel_h == 0.5 and flow_type != 'Kolmogorov':
            axes.axhline(y = 0.5,xmin = 0,
                         xmax = 1,color = 'gray',alpha = 0.4,
                         linestyle = 'dashed')
            axes.set_ylim(-0.6,0.6)
            axes.set_yticks(np.linspace(-0.5,0.5,5))            
        elif channel_h == 0.75 and flow_type != 'Kolmogorov':
            axes.set_ylim(-0.8,0.8)
            axes.set_yticks(np.linspace(-0.75,0.75,7)) 
        elif flow_type == 'Kolmogorov':
            axes.set_ylim(-0.2,2.1)
            axes.set_yticks(np.linspace(0,2,5))
        axes.set_aspect(1*(axes.get_xlim()[1] -axes.get_xlim()[0])/(axes.get_ylim()[1] - axes.get_ylim()[0]))
            
        ### Save figure ###
        filename_png = '{}.png'.format(filename_prefix)
        filename_pdf = '{}.pdf'.format(filename_prefix)
        filename_eps = '{}.eps'.format(filename_prefix)
        
        plt.savefig(os.path.join(output_dir,filename_png),bbox_inches = 'tight',
                    dpi = 200)
        plt.savefig(os.path.join(output_dir,filename_pdf),bbox_inches = 'tight',
                    dpi = 200)
        plt.savefig(os.path.join(output_dir,filename_eps),bbox_inches = 'tight',
                    dpi = 200,format = 'eps')
        plt.show()
