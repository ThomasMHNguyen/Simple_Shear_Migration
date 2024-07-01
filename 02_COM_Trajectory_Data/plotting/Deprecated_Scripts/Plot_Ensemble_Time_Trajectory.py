# -*- coding: utf-8 -*-
"""
FILE NAME:      Plot_Ensemble_Time_Trajectory.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):        A__v01_02_Process_COM_Migration_Data.py

DESCRIPTION:    This script will plot the filament center of mass and true centers
                for each ensemble as a function of time. 

INPUT
FILES(S):       1) .CSV file that contains the center of mass position, true
                center position, and stress values at each time step during 
                the simulation for all ensembles. 

OUTPUT
FILES(S):       1) .PNG file that shows the trajectory of each ensemble simulation.
                1) .PDF file that shows the trajectory of each ensemble simulation.
                1) .EPS file that shows the trajectory of each ensemble simulation.


INPUT
ARGUMENT(S):    1) Input Ensemble File: The true path to the master .CSV file
                that contains the center of mass position, true center position, 
                and stress values at each time step during the simulation for 
                all ensembles. 
                2) Main Output directory: The directory that will house all of the
                output files associated with the analysis; if it doesn't exist,
                it will be created.

CREATED:        22Nov22

MODIFICATIONS
LOG:
22Nov22         1) Migrated code to generate the plots from the original script
                to its own instance here.
19Dec22         2) Modified codes for Kolmogorov flows.

    
            
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.8.8

VERSION:        1.0

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:     N/A

NOTE(S):        N/A

"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


       
def plot_time_displacement_com_data(ensemble_data_df,output_dir):
    """
    This function plots the true center or center of mass trajectory of a 
    filament of all ensembles based on a particular rigidity profile, 
    flow strength value, channel height, starting vertical displacement, 
    steric velocity exponential coefficient, flow type, and whether or not
    the sterics algorithm was used or not.
    
    Inputs:
        
    ensemble_data_df:           DataFrame that contains all ensemble
                                center of mass data.
    output_directory:           Directory where the generated plots will be stored in.
    """
        
    #Group by Rigidity Profile, Mu_bar,Channel Height, Vertical Displacement, Velocity Exponential Coefficient
    exp_groups = ensemble_data_df.groupby(
        by = ['Rigidity Suffix','Mu_bar','Channel Height','Starting Vertical Displacement',  #0-3
              'Poiseuille U Centerline','Kolmogorov Phase Text','Kolmogorov Phase Value','Kolmogorov Frequency', #4-7
              'Steric Velocity Exponential Coefficient','Steric Velocity Gap Criteria', #8-9
              'Sterics Use','Flow Type']) #10-11
    for group in exp_groups.groups.keys():
        group_df = exp_groups.get_group(group)
        rigid,mu_bar,channel_h,vert_displ,u_centerline,k_phase_text,k_phase_val,k_freq = group[:8]
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
        vd_txt = '{:.2f}'.format(vert_displ).replace('.','p')
        ##### Plot All Flows #####
        if flow_type == 'Shear':
            abr_flow_txt = 'SHR'
            filename_prefix = '{}_ST_{}_MB_{}_H_{}_VD_{}'.format(steric_use,
                                                            abr_flow_txt,
                                                            mu_bar_text,
                                                            channel_h_text,
                                                            vd_txt)
            if steric_use == 'Yes':
                figure_title = r"$ y^{{\text{{com}}}}  \text{{vs. t^{{Br}}}} \vert \bar{{\mu}} = {0:.0e}$" "\n" r"$H = {1} \vert y_{{0}} = {2:.2f} \vert U_{{x}} = y$".format(
                mu_bar,channel_h,vert_displ)
            else:
                figure_title = r"$\langle y^{{\text{{com}}}} \rangle \text{{vs. t^{{Br}}}} \vert \bar{{\mu}} = {0:.0e} \vert y_{{0}} = {1:.2f}$" "\n" "U_{{x}} = y".format(mu_bar,vert_displ)
        elif flow_type == 'Poiseuille':
            abr_flow_txt = 'POI'
            filename_prefix = '{}_ST_{}_MB_{}_H_{}_UC_{}_VD_{}'.format(steric_use,
                                                                 abr_flow_txt,
                                                                 mu_bar_text,
                                                                 channel_h_text,
                                                                 U_c_text,
                                                                 vd_txt)
            if steric_use == 'Yes':
                figure_title = r"$ \langle y^{{\text{{com}}}} \rangle$ vs. $ t^{{\text{{Br}}}}\: \vert \: \bar{{\mu}} = {0:.0e} \vert y_{{0}} = {1:.2f}$" "\n" r"$H = {2:.2f} \vert U_{{x}} = {3:.2f}(1-y^{{2}}/{4:.2f}^{{2}})$".format(
                mu_bar,vert_displ,channel_h,u_centerline,channel_h)
            else:
                figure_title = r"$ \langle y^{{\text{{com}}}} \rangle$ vs. $ t^{{\text{{Br}}}}\: \vert \: \bar{{\mu}} = {0:.0e} \vert y_{{0}} = {1:.2f} $" "\n" "U_{{x}} = {2:.2f}(1-y^{{2}}/{3:.2f}^{{2}})$".format(
                    mu_bar,vert_displ,u_centerline,channel_h)
        elif flow_type == 'Kolmogorov':
            abr_flow_txt = 'KMG'
            filename_prefix = '{}_ST_{}_MB_{}_H_{}_FR_{}_PH_{}_VD_{}'.format(steric_use,
                                                                       abr_flow_txt,
                                                                       mu_bar_text,
                                                                       channel_h_text,
                                                                       int(k_freq),
                                                                       k_phase_text,
                                                                       vd_txt)
        
            if steric_use == 'No':
                figure_title = r"$\langle y^{{\text{{com}}}} \rangle \text{{vs. t^{{Br}}}} \vert \bar{{\mu}} = {0:.0e} y_{{0}} = {1:.2f} $" "\n" r"$H = {2:.2f} \vert U_{{x}} = \text{{sin}}\left({3} \times {4:.0f} y\right)$".format(
                mu_bar,vert_displ,channel_h,k_phase_text,k_freq)
        
        show_legend = True
        ##### Center of Mass #####
        fig,axes = plt.subplots(figsize = (7,7))
        sns.lineplot(y = 'Center of Mass-y',x = 'Brownian Time',hue = 'Rep Number',
                              palette = 'bright',data = group_df,linewidth = 1.5,
                              legend = show_legend,ax = axes)
        axes.ticklabel_format(axis="x", style="sci", scilimits=(-2,-2))
        axes = plt.gca()
        axes.set_title(figure_title,fontsize = 15,pad = 13)
        axes.set_xlabel(r"$t^{{\text{{Br}}}} \times 10^{{-2}}$",fontsize = 13,labelpad = 13)
        axes.set_ylabel(r"$y_{\text{{com}}}$",fontsize = 13,labelpad = 13)
        axes.tick_params(axis='both', which='major', labelsize=11,direction = 'in')
        axes.xaxis.offsetText.set_fontsize(0)

        ### Set limits for time display ###
        if group_df['Brownian Time'].max() == 5e-2:
            axes.set_xlim(-1e-6,5.01e-2)
            axes.set_xticks(np.linspace(0,5e-2,6))
        elif group_df['Brownian Time'].max() == 1e-1:
            axes.set_xlim(-5e-6,1e-1)
            axes.set_xticks(np.linspace(0,1e-1,6))
        elif group_df['Brownian Time'].max() == 1.5e-1:
            axes.set_xlim(-1e-5,1.6e-1)
            axes.set_xticks(np.linspace(0,1.5e-1,4))
        elif group_df['Brownian Time'].max() == 2e-1:
            axes.set_xlim(-5e-6,2.1e-1)
            axes.set_xticks(np.linspace(0,2e-1,5))

        ### Set limits for channel height ###
        if channel_h == 0.25 and flow_type != 'Kolmogorov':
            axes.set_ylim(-0.3,0.3)
            axes.set_yticks(np.linspace(-0.25,0.25,3))     
        if channel_h == 0.5 and flow_type != 'Kolmogorov':
            axes.set_ylim(-0.6,0.6)
            axes.set_yticks(np.linspace(-0.5,0.5,5))            
        elif channel_h == 0.75 and flow_type != 'Kolmogorov':
            axes.set_ylim(-0.8,0.8)
            axes.set_yticks(np.linspace(-0.75,0.75,7)) 
        elif flow_type == 'Kolmogorov':
            axes.set_ylim(-0.2,2.1)
            axes.set_yticks(np.linspace(0,2,5))
            
        if show_legend:
            axes.legend(loc = 'lower right',prop={'size': 11},title= "Ensemble Number",title_fontsize = 13)
        axes.set_aspect((np.diff(axes.get_xlim()))/(np.diff(axes.get_ylim())))

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
        
def plot_time_displacement_tc_data(ensemble_data_df,output_dir):
    """
    This function plots the true center or center of mass trajectory of a 
    filament of all ensembles based on a particular rigidity profile, 
    flow strength value, channel height, starting vertical displacement, 
    steric velocity exponential coefficient, flow type, and whether or not
    the sterics algorithm was used or not.
    
    Inputs:
        
    ensemble_data_df:           DataFrame that contains all ensemble
                                center of mass data.
    output_directory:           Directory where the generated plots will be stored in.
    """
        
    #Group by Rigidity Profile, Mu_bar,Channel Height, Vertical Displacement, Velocity Exponential Coefficient
    exp_groups = ensemble_data_df.groupby(
        by = ['Rigidity Suffix','Mu_bar','Channel Height','Starting Vertical Displacement',  #0-3
              'Poiseuille U Centerline','Kolmogorov Phase Text','Kolmogorov Phase Value','Kolmogorov Frequency', #4-7
              'Steric Velocity Exponential Coefficient','Steric Velocity Gap Criteria', #8-9
              'Sterics Use','Flow Type']) #10-11
    for group in exp_groups.groups.keys():
        group_df = exp_groups.get_group(group)
        rigid,mu_bar,channel_h,vert_displ,u_centerline,k_phase_text,k_phase_val,k_freq = group[:8]
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
        vd_txt = '{:.2f}'.format(vert_displ).replace('.','p')
        ##### Plot All Flows #####
        if flow_type == 'Shear':
            abr_flow_txt = 'SHR'
            filename_prefix = 'YTC_{}_ST_{}_MB_{}_H_{}_VD_{}'.format(steric_use,
                                                            abr_flow_txt,
                                                            mu_bar_text,
                                                            channel_h_text,
                                                            vd_txt)
            if steric_use == 'Yes':
                figure_title = r"$ y^{{\text{{com}}}}  \text{{vs. t^{{Br}}}} \vert \bar{{\mu}} = {0:.0e}$" "\n" r"$H = {1} \vert y_{{0}} = {2:.2f} \vert U_{{x}} = y$".format(
                mu_bar,channel_h,vert_displ)
            else:
                figure_title = r"$\langle y^{{\text{{com}}}} \rangle \text{{vs. t^{{Br}}}} \vert \bar{{\mu}} = {0:.0e} \vert y_{{0}} = {1:.2f}$" "\n" "U_{{x}} = y".format(mu_bar,vert_displ)
        elif flow_type == 'Poiseuille':
            abr_flow_txt = 'POI'
            filename_prefix = 'YTC_{}_ST_{}_MB_{}_H_{}_UC_{}_VD_{}'.format(steric_use,
                                                                 abr_flow_txt,
                                                                 mu_bar_text,
                                                                 channel_h_text,
                                                                 U_c_text,
                                                                 vd_txt)
            if steric_use == 'Yes':
                figure_title = r"$ \langle y^{{\text{{com}}}} \rangle$ vs. $ t^{{\text{{Br}}}}\: \vert \: \bar{{\mu}} = {0:.0e} \vert y_{{0}} = {1:.2f}$" "\n" r"$H = {2:.2f} \vert U_{{x}} = {3:.2f}(1-y^{{2}}/{4:.2f}^{{2}})$".format(
                mu_bar,vert_displ,channel_h,u_centerline,channel_h)
            else:
                figure_title = r"$ \langle y^{{\text{{com}}}} \rangle$ vs. $ t^{{\text{{Br}}}}\: \vert \: \bar{{\mu}} = {0:.0e} \vert y_{{0}} = {1:.2f} $" "\n" "U_{{x}} = {2:.2f}(1-y^{{2}}/{3:.2f}^{{2}})$".format(
                    mu_bar,vert_displ,u_centerline,channel_h)
        elif flow_type == 'Kolmogorov':
            abr_flow_txt = 'KMG'
            filename_prefix = 'YTC_{}_ST_{}_MB_{}_H_{}_FR_{}_PH_{}_VD_{}'.format(steric_use,
                                                                       abr_flow_txt,
                                                                       mu_bar_text,
                                                                       channel_h_text,
                                                                       int(k_freq),
                                                                       k_phase_text,
                                                                       vd_txt)
        
            if steric_use == 'No':
                figure_title = r"$\langle y^{{\text{{com}}}} \rangle \text{{vs. t^{{Br}}}} \vert \bar{{\mu}} = {0:.0e} y_{{0}} = {1:.2f} $" "\n" r"$H = {2:.2f} \vert U_{{x}} = \text{{sin}}\left({3} \times {4:.0f} y\right)$".format(
                mu_bar,vert_displ,channel_h,k_phase_text,k_freq)
        
        show_legend = True
        ##### Center of Mass #####
        fig,axes = plt.subplots(figsize = (7,7))
        sns.lineplot(y = 'True Center-y',x = 'Brownian Time',hue = 'Rep Number',
                              palette = 'bright',data = group_df,linewidth = 1.5,
                              legend = show_legend,ax = axes)
        axes.ticklabel_format(axis="x", style="sci", scilimits=(-2,-2))
        axes = plt.gca()
        axes.set_title(figure_title,fontsize = 15,pad = 13)
        axes.set_xlabel(r"$t^{{\text{{Br}}}} \times 10^{{-2}}$",fontsize = 13,labelpad = 13)
        axes.set_ylabel(r"$y_{\text{{tc}}}$",fontsize = 13,labelpad = 13)
        axes.tick_params(axis='both', which='major', labelsize=11,direction = 'in')
        axes.xaxis.offsetText.set_fontsize(0)
    
        ### Set limits for time display ###
        if group_df['Brownian Time'].max() == 5e-2:
            axes.set_xlim(-1e-6,5.01e-2)
            axes.set_xticks(np.linspace(0,5e-2,6))
        elif group_df['Brownian Time'].max() == 1e-1:
            axes.set_xlim(-5e-6,1e-1)
            axes.set_xticks(np.linspace(0,1e-1,6))
        elif group_df['Brownian Time'].max() == 1.5e-1:
            axes.set_xlim(-1e-5,1.6e-1)
            axes.set_xticks(np.linspace(0,1.5e-1,4))
        elif group_df['Brownian Time'].max() == 2e-1:
            axes.set_xlim(-5e-6,2.1e-1)
            axes.set_xticks(np.linspace(0,2e-1,5))
    
        ### Set limits for channel height ###
        if channel_h == 0.25 and flow_type != 'Kolmogorov':
            axes.set_ylim(-0.3,0.3)
            axes.set_yticks(np.linspace(-0.25,0.25,3))     
        if channel_h == 0.5 and flow_type != 'Kolmogorov':
            axes.set_ylim(-0.6,0.6)
            axes.set_yticks(np.linspace(-0.5,0.5,5))            
        elif channel_h == 0.75 and flow_type != 'Kolmogorov':
            axes.set_ylim(-0.8,0.8)
            axes.set_yticks(np.linspace(-0.75,0.75,7)) 
        elif flow_type == 'Kolmogorov':
            axes.set_ylim(-0.2,2.1)
            axes.set_yticks(np.linspace(0,2,5))
            
        if show_legend:
            axes.legend(loc = 'lower right',prop={'size': 11},title= "Ensemble Number",title_fontsize = 13)
        axes.set_aspect((np.diff(axes.get_xlim()))/(np.diff(axes.get_ylim())))
    
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
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_directory",
                        help="Specify the path to the directory that contains this script and all other relevant scripts",
                    type = str)
    parser.add_argument("input_ensemble_file",
                        help="Specify the Absolute Path to the CSV file that contains the ensemble center data",
                    type = str,default = None)
    parser.add_argument("output_directory",
                        help = "Specify the directory where the resulting plots will be saved in",
                        type = str,default = None)
    args = parser.parse_args()
    os.chdir(args.project_directory)
    from misc.create_dir import create_dir
    
    create_dir(args.output_directory)
    ensemble_data_df = pd.read_csv(args.input_ensemble_file,index_col = 0,header = 0)
    
    plot_time_displacement_com_data(ensemble_data_df,args.output_directory)
    plot_time_displacement_tc_data(ensemble_data_df,args.output_directory)
    
    
    
