# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:07:39 2023

@author: super
"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class sim_data():
    def __init__(self,poi_dir_1,poi_dir_2,output_dir):
        from calculations.adjust_position_data import adjust_position_data
        
        self.poi_data_dir_1 = poi_dir_1
        self.poi_data_dir_2 = poi_dir_2
        self.output_dir = output_dir
        
        self.position_data_1 = np.load(os.path.join(self.poi_data_dir_1,'filament_allstate.npy'))
        self.time_data_1 = np.load(os.path.join(self.poi_data_dir_1,'filament_time_vals_all.npy'))
        self.param_data_1 = pd.read_csv(os.path.join(self.poi_data_dir_1,'parameter_values.csv'),
                                        index_col = 0,header = 0)
        
        self.position_data_2 = np.load(os.path.join(self.poi_data_dir_2,'filament_allstate.npy'))
        self.time_data_2 = np.load(os.path.join(self.poi_data_dir_2,'filament_time_vals_all.npy'))
        self.param_data_2 = pd.read_csv(os.path.join(self.poi_data_dir_2,'parameter_values.csv'),
                                        index_col = 0,header = 0)
        
        self.adj_position_data_1 = adjust_position_data(position_data = self.position_data_1,
                                                        adj_centering = True,
                                                        adj_translation = False,
                                                        transl_val = None)
        self.adj_position_data_2 = adjust_position_data(position_data = self.position_data_2,
                                                        adj_centering = True,
                                                        adj_translation = False,
                                                        transl_val = None)

    def create_dir(self):
        from misc.create_dir import create_dir
        create_dir(self.output_dir)
    
    def plot_snapshots(self):
        self.create_dir()
        file_name = 'flipping_snapshots_toc'
        
        cm = 1/2.54  # centimeters in inches
        
        timepoints_1 = np.array([1e-3,1.2e-3,1.4e-3,1.45e-3,1.9e-3]) #Need 5 values
        timepoints_2 = np.array([1.87e-2,1.885e-2,1.889e-2,1.895e-2,1.93e-2]) #Need 5 values
        fig,axes = plt.subplots(figsize=(8*cm, 4*cm),ncols = 5,nrows = 2)
        # fig,axes = plt.subplots(ncols = 5,nrows = 2)
        fig.subplots_adjust(wspace=-0.4, hspace=0.3)
        # fig.set_size_cm(8,4)
        #Plot data #
        for n_col,ax_col in enumerate(axes[0,:]):
            time_idx_1 = np.where(self.time_data_1 == timepoints_1[n_col])[0][0]
            
            
            current_position_1 = self.adj_position_data_1[...,time_idx_1]
            
            
            ax_col.plot(current_position_1[:,0],current_position_1[:,1],
                        color = '#C96480',linewidth = 1.5)
        for n_col,ax_col in enumerate(axes[1,:]):
            time_idx_2 = np.where(self.time_data_2 == timepoints_2[n_col])[0][0]
            current_position_2 = self.adj_position_data_2[...,time_idx_2]
            ax_col.plot(current_position_2[:,0],current_position_2[:,1],
                        color = '#083D77',linewidth = 1.5)
            
        ### Annotate to show progression of filament shape over time ###
        # axes[0][0].annotate(text = "",xy = (4.7,-0.68),
        #                   xytext = (-0.5,-0.68),arrowprops=dict(arrowstyle='->', connectionstyle='arc,rad=0', 
        #                             mutation_scale = 14,color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2),annotation_clip = False)
        # axes[1][0].annotate(text = "",xy = (4.7,-0.68),
        #                   xytext = (-0.5,-0.68),arrowprops=dict(arrowstyle='->', connectionstyle='arc,rad=0', 
        #                             mutation_scale = 14,color='black',linewidth = 1.5,shrinkA=0.2, shrinkB=0.2),annotation_clip = False)
        # axes[0][2].text(x = -0.30,y = -0.92,s = "Time",fontsize = 6)
        # axes[1][2].text(x = -0.30,y = -0.92,s = "Time",fontsize = 6)
        axes[0][2].text(x = -0.80,y = 0.60,s = "Wall Exclusion",fontsize = 7)
        axes[1][2].text(x = -1.10,y = 0.60,s = "Shear-induced Drift",fontsize = 7)
            
        ### Plot Velocity profile in main subplot ###
        channel_height,u_centerline = 0.5,1
        y = np.linspace(-1*0.5,1*channel_height,201)
        ux = (u_centerline*0.5*(1-(y**2)/(channel_height**2)))- 0.5
        axes[0][0].plot(ux,y,color = 'blue',linewidth = 1.1,linestyle = 'solid',alpha = 0.2)
        axes[1][0].plot(ux,y,color = 'blue',linewidth = 1.1,linestyle = 'solid',alpha = 0.2)
        slicing_factor = 40
        y_subset = y[::slicing_factor].copy()
        x_subset = ux[::slicing_factor].copy()
        quiv_x = np.zeros_like(y_subset) - 0.5
        quiv_y = y_subset.copy()
        quiv_ar_x = x_subset.copy() + 0.5
        quiv_ar_y = np.zeros_like(y_subset)
        axes[0][0].quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                        scale_units='xy', scale=1,color = 'blue',headwidth = 12,
                        headlength = 10,alpha = 0.2)
        axes[1][0].quiver(quiv_x,quiv_y,quiv_ar_x,quiv_ar_y,angles='xy', 
                        scale_units='xy', scale=1,color = 'blue',headwidth = 12,
                        headlength = 10,alpha = 0.2)
            
        ### Format snapshots of filament vaulting axes ###  
        for n_row,ax_row in enumerate(axes):
            for n_col,ax_col in enumerate(ax_row):
                ax_col.tick_params(axis='both', which='major', direction = 'in',labelsize=4,pad = 5)
                ax_col.set_ylim(-0.55,0.55)
                ax_col.set_yticks(np.linspace(-0.5,0.5,3))
                ax_col.set_xlim(-0.55,0.55)
                ax_col.set_xticks(np.linspace(-0.5,0.5,5))
                ax_col.get_xaxis().set_visible(False)
                ax_col.axhline(y = 0.5,xmin = 0,
                              xmax = 1,color = 'gray',alpha = 0.4,
                              linestyle = 'dashed')
                ax_col.axhline(y = -0.5,xmin = 0,
                              xmax = 1,color = 'gray',alpha = 0.4,
                              linestyle = 'dashed')
                
                if n_col == 0:
                    # ax_col.axvline(x = -0.5,ymin = 0.5,ymax = 1,color = 'black',linewidth = 0.8)
                    ax_col.set_ylabel(r"y",fontsize = 5,labelpad = 1)
                    ax_col.spines['right'].set_visible(False)
                else:
                    ax_col.get_yaxis().set_visible(False)
                    ax_col.spines['left'].set_visible(False)
                    if n_col != 4:
                        ax_col.spines['right'].set_visible(False)
                ax_col.set_aspect(np.diff(ax_col.get_xlim())/(np.diff(ax_col.get_ylim())))
                ax_col.set_rasterized(True)
                
                
        filename_png = '{}.png'.format(file_name)
        filename_pdf = '{}.pdf'.format(file_name)
        filename_eps = '{}.eps'.format(file_name)
        
        plt.savefig(os.path.join(self.output_dir,filename_png),bbox_inches = 'tight',
                    format = 'png',dpi = 600)
        plt.savefig(os.path.join(self.output_dir,filename_pdf),bbox_inches = 'tight',
                    format = 'pdf',dpi = 600)
        plt.savefig(os.path.join(self.output_dir,filename_eps),bbox_inches = 'tight',
                    dpi = 600,format = 'eps')
        
        plt.show()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_directory",
                        help="Specify the path to the directory that contains this script and all other relevant scripts",
                    type = str)
    parser.add_argument("input_poiseuille_directory_1",
                        help = "Specify the path to the directories where the filament data corresponding (1 of 2)",
                        type = str)
    parser.add_argument("input_poiseuille_directory_2",
                        help = "Specify the path to the directories where the filament data corresponding (2 of 2)",
                        type = str)
    parser.add_argument("output_directory",
                        help="Specify the parent directory of the Migration Data",
                    type = str)
    args = parser.parse_args(['C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Scripts//04_Table_of_Contents_Fig//',
                              'C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Remote_Data//Poiseuille_Flow_Walls//VD_0p45//K_constant_UC_1p00//MB_10000//R_4//',
                              'C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Remote_Data//Poiseuille_Flow_Walls//VD_0p05//K_constant_UC_1p00//MB_50000//R_1//',
                              'C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Scripts//04_Table_of_Contents_Fig//Fig//'])
    os.chdir(args.project_directory)
    
    snapshot_data = sim_data(poi_dir_1 = args.input_poiseuille_directory_1,
             poi_dir_2 = args.input_poiseuille_directory_2,
             output_dir = args.output_directory)
    snapshot_data.plot_snapshots()
    
    