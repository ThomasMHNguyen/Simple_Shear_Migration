# -*- coding: utf-8 -*-
"""
FILE NAME:      Plot_Ensemble_COM_Distribution.py
    
COMPLEMENTARY
SCRIPT(S)/
FILE(S):        A__v01_02_Process_COM_Migration_Data.py

DESCRIPTION:    This script will plot the probability distribution of all filament
                ensembles' COM at various time points during a simulation. 

INPUT
FILES(S):       1) .CSV file that contains the center of mass position, true
                center position, and stress values at each time step during 
                the simulation for all ensembles. 

OUTPUT
FILES(S):       1) .PNG file that shows the probability distribution of the COM
                at various timepoints.
                2) .PDF file that shows the probability distribution of the COM
                at various timepoints.
                3) .EPS file that shows the probability distribution of the COM 
                at various timepoints.


INPUT
ARGUMENT(S):    1) Input Ensemble File: The true path to the master .CSV file
                that contains the center of mass position, true center position, 
                and stress values at each time step during the simulation for 
                all ensembles. 
                2) Main Output directory: The directory that will house all of the
                output files associated with the analysis; if it doesn't exist,
                it will be created.

CREATED:        22Jan23

MODIFICATIONS
LOG:


    
            
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

import re, os,argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})

      
def plot_com_distribution(ensemble_data_df,time_vals,time_vals_text,output_dir):
    
    #group data by various parameters
    exp_groups = ensemble_data_df.groupby(
        by = ['Rigidity Suffix','Mu_bar','Channel Height', #0-2
              'Poiseuille U Centerline','Kolmogorov Phase Text','Kolmogorov Phase Value','Kolmogorov Frequency', #3-6
              'Steric Velocity Exponential Coefficient','Steric Velocity Gap Criteria', #7-8
              'Sterics Use','Flow Type']) #9-10
    for group in exp_groups.groups.keys():
        group_df = exp_groups.get_group(group)
        rigid,mu_bar,channel_h,u_centerline,k_phase_text,k_phase_val,k_freq = group[:7]
        steric_use,flow_type = group[-2:]
        
        time_val_0_data = group_df[group_df['Brownian Time'] == time_vals[0]]
        time_val_1_data = group_df[group_df['Brownian Time'] == time_vals[1]]
        time_val_2_data = group_df[group_df['Brownian Time'] == time_vals[2]]
        time_val_3_data = group_df[group_df['Brownian Time'] == time_vals[3]]
        time_val_4_data = group_df[group_df['Brownian Time'] == time_vals[4]]
        
        
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
        
        
        fig,axes = plt.subplots(ncols = 6,figsize = (11,7),layout = 'constrained')
        
            
        ##### Plot non-Kolmogorov flows-COM #####
        if k_freq == 0:
            
            
            if flow_type == 'Poiseuille':
                flow_profile = r"U_{{x}} = {0}(1-y^{{2}}/{1}^{{2}})".format(u_centerline,channel_h)
                abr_flow_txt = 'POI'
                filename_prefix = '{}_ST_{}_MB_{}_H_{}_UC_{}'.format(steric_use,
                                                                     abr_flow_txt,
                                                                     mu_bar_text,
                                                                     channel_h_text,
                                                                     U_c_text)
                ### Flow profile ###
                y = np.linspace(-1*channel_h,1*channel_h,200)
                x = u_centerline*(1-(y**2)/(channel_h**2))
                dx_dy = -2*u_centerline*y/channel_h**2
            elif flow_type == 'Shear':
                flow_profile = r"U_{x} = y"
                abr_flow_txt = 'SHR'
                filename_prefix = '{}_ST_{}_MB_{}_H_{}'.format(steric_use,
                                                                abr_flow_txt,
                                                                mu_bar_text,
                                                                channel_h_text)
                ### Flow profile ###
                y = np.linspace(-1*channel_h,1*channel_h,200)
                x = y
                dx_dy = 1
              
            ### Plot COM distribution ###
            sns.histplot(data = time_val_0_data,
                            y = 'Center of Mass-y',stat = 'probability',
                            fill = 'True',alpha = 0.2,linewidth = 0,
                            # binwidth = (time_val_0_data['Center of Mass-y'].max() - time_val_0_data['Center of Mass-y'].min())/\
                            #     time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                            bins = time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                            kde = True,
                            ax = axes[1],legend = False,color = "#7209B7")
                                
            
            sns.histplot(data = time_val_1_data,
                        y = 'Center of Mass-y',stat = 'probability',
                        fill = 'True',alpha = 0.2,linewidth = 0,
                        # bins = 9,
                        kde = True,
                        ax = axes[2],legend = False,color = "#7209B7")
            sns.histplot(data = time_val_2_data,
                        y = 'Center of Mass-y',stat = 'probability',
                        fill = 'True',alpha = 0.2,linewidth = 0,
                        # bins = 9,
                        kde = True,
                        ax = axes[3],legend = False,color = "#7209B7")
            sns.histplot(data = time_val_3_data,
                        y = 'Center of Mass-y',stat = 'probability',
                        fill = 'True',alpha = 0.2,linewidth = 0,
                        # bins = 9,
                        kde = True,
                        ax = axes[4],legend = False,color = "#7209B7")
            sns.histplot(data = time_val_4_data,
                        y = 'Center of Mass-y',stat = 'probability',
                        fill = 'True',alpha = 0.2,linewidth = 0,
                        # bins = 9,
                        kde = True,
                        ax = axes[5],legend = False,color = "#7209B7")
        ##### Plot Kolmogorov flows-COM #####
        else:
            if k_phase_text != 'Pi':
                flow_profile = r"U_{{x}} =$ sin$\left({0:.0f} \times {1:.2f} y\right)".format(k_freq,k_phase_text)
            else:
                flow_profile = r"U_{{x}} =$ sin$\left({0:.0f} \times \pi y\right)".format(k_freq)
            abr_flow_txt = 'KMG'
            filename_prefix = '{}_ST_{}_MB_{}_H_{}_FR_{}_PH_{}'.format(steric_use,
                                                                       abr_flow_txt,
                                                                       mu_bar_text,
                                                                       channel_h_text,
                                                                       int(k_freq),
                                                                       k_phase_text)
            sns.histplot(data = time_val_0_data,
                            y = 'Center of Mass-y',stat = 'density',
                            fill = 'True',alpha = 0.2,linewidth = 0.0,
                            # binwidth = (time_val_0_data['Center of Mass-y'].max() - time_val_0_data['Center of Mass-y'].min())/\
                            #     time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                            bins = time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                            kde = True,
                            ax = axes[1],legend = False,color = "#7209B7")
                                
            
            sns.histplot(data = time_val_1_data,
                        y = 'Center of Mass-y',stat = 'density',
                        fill = 'True',alpha = 0.2,linewidth = 0,
                        binwidth = (time_val_0_data['Center of Mass-y'].max() - time_val_0_data['Center of Mass-y'].min())/\
                            time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                        # binwidth = 0.12,
                        kde = True,
                        ax = axes[2],legend = False,color = "#7209B7")
            sns.histplot(data = time_val_2_data,
                        y = 'Center of Mass-y',stat = 'density',
                        fill = 'True',alpha = 0.2,linewidth = 0,
                        binwidth = (time_val_0_data['Center of Mass-y'].max() - time_val_0_data['Center of Mass-y'].min())/\
                            time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                        # binwidth = 0.12,
                        kde = True,
                        ax = axes[3],legend = False,color = "#7209B7")
            sns.histplot(data = time_val_3_data,
                        y = 'Center of Mass-y',stat = 'density',
                        fill = 'True',alpha = 0.2,linewidth = 0,
                        binwidth = (time_val_0_data['Center of Mass-y'].max() - time_val_0_data['Center of Mass-y'].min())/\
                            time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                        # binwidth = 0.12,
                        kde = True,
                        ax = axes[4],legend = False,color = "#7209B7")
            sns.histplot(data = time_val_4_data,
                        y = 'Center of Mass-y',stat = 'density',
                        fill = 'True',alpha = 0.2,linewidth = 0,
                        binwidth = (time_val_0_data['Center of Mass-y'].max() - time_val_0_data['Center of Mass-y'].min())/\
                            time_val_0_data['Starting Vertical Displacement'].unique().shape[0],
                        # binwidth = 0.12,
                        kde = True,
                        ax = axes[5],legend = False,color = "#7209B7")
            
            
            ### Flow profile ###
            y = np.linspace(-2.1,3.1,200)
            x = np.sin(k_freq*k_phase_val*y)
            dx_dy = k_freq*k_phase_val*np.cos(k_freq*k_phase_val*y)
            
            
        ### Plot Velocity Profiles & Starting Points ###
        axes[0].plot(x,y,color = 'blue',linewidth = 1.5,linestyle = 'solid',
                        label = r'$U_{x}$')
        
        if flow_type == 'Kolmogorov':
            axes[0].plot(dx_dy,y,color = 'black',linewidth = 1.3,linestyle = 'dashed',
                            label = r'$\partial U_{x}/ \partial y$',alpha = 0.7)

        slicing_factor = 10
        y_subset = y[::slicing_factor].copy()
        x_subset = x[::slicing_factor].copy()
        quiv_x = np.zeros_like(y_subset)
        quiv_y = y_subset.copy()
        quiv_ar_x = x_subset.copy()
        quiv_ar_y = np.zeros_like(y_subset)
        axes[0].quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                       scale_units='xy', scale=1,color = 'blue',headwidth = 15,
                       headlength = 7)
        
        subfigure_text = [r"\textbf{(a)}",r"\textbf{(b)}",r"\textbf{(c)}",r"\textbf{(d)}",r"\textbf{(e)}",r"\textbf{(f)}"]
        if flow_type == 'Poiseuille' or flow_type == 'Shear':
        ### Formatting on plots ###
            for n,ax in enumerate(axes):
                    if n > 0:
                        # ax_col.set_xlabel(r"Density",fontsize = 25,labelpad = 15)
                        ax.set_ylabel(r"$y^{\text{com}}$",fontsize = 13,labelpad = 5)
                        
                        ax.set_ylim(-0.05,np.round(1.1*channel_h,2))
                        ax.set_xlim(0,0.55)
                        ax.set_yticks(np.linspace(0,0.5,6))
                        ax.set_xticks(np.linspace(0,0.5,6))
                        ax.tick_params(axis='both', which='major',direction = 'in', labelsize=11,pad = 5)
                        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i) % 2 != 0]
                        ax.set_title(r"$t^{{\text{{Br}}}} = {0}$".format(time_vals_text[n - 1]),
                                            fontsize = 11,pad = 5)
                        ax.set_aspect(2*np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
                        
                    elif n == 0: #Velocity profile distribution
                        # ax.set_xlabel(r"$U_{x}$ or $\partial U_{x}/ \partial y$",
                                            # fontsize = 11,labelpad = 5)
                        ax.set_xlabel(r"$U_{x} (y)$",
                                            fontsize = 11,labelpad = 5)                  
                        ax.set_ylabel(r"$y$",fontsize = 13,labelpad = 5)
                        ax.set_ylim(-0.05,np.round(1.1*channel_h,2))
                        ax.set_yticks(np.linspace(0,0.5,6))
                        if flow_type == 'Poiseuille':
                            ax.set_xlim(-0.1,1.6)
                            ax.set_xticks(np.linspace(-0,1.5,6))
                            ax.hlines(y = channel_h,xmin =-0.1,xmax = 1.6,
                                     color = 'black',linestyle = 'dashed',linewidth = 1.2,alpha = 0.7)
                            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i) % 2 != 0]
                        elif flow_type == 'Shear':
                            ax.set_xlim(-0.05,0.6)
                            ax.set_xticks(np.linspace(-0,0.5,6))
                            ax.hlines(y = channel_h,xmin =-0.05,xmax = 0.6,
                                     color = 'black',linestyle = 'dashed',linewidth = 1.2,alpha = 0.7)
                            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i) % 2 != 0]
                        ax.tick_params(axis='both', which='major',direction = 'in', labelsize=11,pad = 5)
                        
                        ax.set_title(r"Velocity Distribution",
                                            fontsize = 11,pad = 5)
                        # ax.legend(loc = 'center left',bbox_to_anchor = (-0.03,1.14),fontsize = 10)
                        ax.set_aspect(2*np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
                    axes[0].vlines(x = 0,ymin = axes[0].get_ylim()[0],ymax = channel_h,color = 'gray',alpha = 0.7)
             
        elif flow_type == 'Kolmogorov':
        ### Formatting on plots ###
            for n,ax in enumerate(axes):
                    if n > 0:
                        ax.text(x = -0.55,y = 2.5,s = subfigure_text[n],fontsize = 14)
                        ax.set_xlabel(r"Density",fontsize = 13,labelpad = 3)
                        ax.set_ylabel(r"$y^{\text{com}}$",fontsize = 13,labelpad = 2)
                        ax.tick_params(axis='both', which='major',direction = 'in', labelsize=11,pad = 5)
                        ax.set_ylim(-0.6,2.6)
                        ax.set_xlim(0,1.01)
                        ax.set_yticks(np.linspace(-0.5,2.5,7))
                        ax.set_xticks(np.linspace(0,1,5))
                        # ax.set_title(r"$t^{{\text{{Br}}}} = {0}$".format(time_vals_text[n - 1]),
                        #                     fontsize = 11,pad = 5)
                        if n  == 2:
                            ax.set_title(r"$t^{{\text{{Br}}}} = {0:.3f}$".format(time_vals[n - 1]),
                                                fontsize = 11,pad = 5)
                        else:
                            ax.set_title(r"$t^{{\text{{Br}}}} = {0:.2f}$".format(time_vals[n - 1]),
                                                fontsize = 11,pad = 5)
                        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i) % 2 != 0]
                        ax.set_aspect(2*np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
                        
                    elif n == 0: #Velocity profile distribution
                        ax.text(x = -7.65,y = 2.5,s = subfigure_text[n],fontsize = 14)
                        ax.set_xlabel(r"$U_{x} (y) \text{ or } \partial U_{x}/ \partial y (y)$",
                                            fontsize = 13,labelpad = 3)
                        ax.set_ylabel(r"$y$",fontsize = 13,labelpad = 2)
                        ax.set_ylim(-0.6,2.6)
                        ax.set_yticks(np.linspace(-0.5,2.5,7))
                        ax.set_xlim(-3.5,3.5)
                        ax.set_xticks(np.linspace(-3,3,7))
                        ax.tick_params(axis='both', which='major',direction = 'in', labelsize=11,pad = 5)
                        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if (i-1) % 2 != 0]
                        # ax.legend(loc = 'center left',bbox_to_anchor = (-0.03,1.14),fontsize = 10)
                        ax.set_aspect(2*np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
                        axes[0].vlines(x = 0,ymin = axes[0].get_ylim()[0],ymax = axes[0].get_ylim()[1],color = 'gray',alpha = 0.7)
        # fig.suptitle(r"$\bar{{\mu}} = {0:.1e} \vert {1} \vert H = {2:.2f}$".format(mu_bar,flow_profile,channel_h),size = 15,y = 0.75)
                        
        ### Save figure-COM ###
            
        filename_png = '{}.png'.format(filename_prefix)
        filename_pdf = '{}.pdf'.format(filename_prefix)
        filename_eps = '{}.eps'.format(filename_prefix)
        
        # fig.savefig(os.path.join(output_dir,filename_png),bbox_inches = 'tight',
        #             dpi = 200)
        # fig.savefig(os.path.join(output_dir,filename_pdf),bbox_inches = 'tight',
        #             dpi = 200)
        # fig.savefig(os.path.join(output_dir,filename_eps),bbox_inches = 'tight',
        #             dpi = 200,format = 'eps')
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
    parser.add_argument("Time_1",
                        help = 'Specify the 1st time you want to plot the COM distribution data for',
                        type = float)
    parser.add_argument("Time_2",
                        help = 'Specify the 2nd time you want to plot the COM distribution data for',
                        type = float)
    parser.add_argument("Time_3",
                        help = 'Specify the 3rd time you want to plot the COM distribution data for',
                        type = float)
    parser.add_argument("Time_4",
                        help = 'Specify the 4th time you want to plot the COM distribution data for',
                        type = float)
    parser.add_argument("Time_5",
                        help = 'Specify the 5th time you want to plot the COM distribution data for',
                        type = float)
    parser.add_argument("output_directory",
                        help = "Specify the directory where the resulting plots will be saved in",
                        type = str,default = None)
    args = parser.parse_args()
    os.chdir(args.project_directory)
    from misc.create_dir import create_dir
    
    ensemble_data_df = pd.read_csv(args.input_ensemble_file,index_col = 0,header = 0)
    all_time_vals = np.array([args.Time_1,args.Time_2,args.Time_3,
                              args.Time_4,args.Time_5])
    create_dir(args.output_directory)
    
    plot_com_distribution(ensemble_data_df,all_time_vals,args.output_directory)
            