# -*- coding: utf-8 -*-
"""
FILE NAME:      Compare_Poiseuille_Shear_Data.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        create_dir.py; adjust_center_of_position.py; center_of_mass.py;
                plot_com_time_curves.py; plot_net_com_comp.py

DESCRIPTION:    For a given vertical displacement of a filament in a non-Brownian 
                simulation in Poiseuille flow, this script will find the 
                equivalent data in Shear flow according to the following equation:
                    
                $\bar{\mu}^{\text{shear}}_{\text{eff}} = 2\bar{\mu}^{\text{Poiseuille}}y_{0}/H^{2}$
                
                Here, H is the channel height and is equivalent to 0.5L and 
                the mu_bar of Poiseuille flow is equivalent to 10^5. Once the 
                corresponding data has been found, this script will read the position
                data in to calculate the center of mass (com) and generate plots that
                compare the time course curves and net com between the different
                flows and flip direction/displacement type of the filament.

INPUT
FILES(S):       This script will find the input files needed to run the script. 
                No input files are specified as an argument.

OUTPUT
FILES(S):       
    
1)              .PNG/.PDF/.EPS file that shows center of mass curves for 
                Poiseuille for 5 equivalent starting vertical displacements 
                and the equivalent mu_bar in shear flow.
2)              .PNG/.PDF/.EPS file that shows net com displacement as a function
                of the effective shear flow mu_bar. 

INPUT
ARGUMENT(S):    
    
1)              Project Directory: The directory that contains all of the 
                complementary scripts needed to run the analysis.
2)              Main Input Poiseuille directory: The directory that will houses 
                all of the Poseuille flow simulation data.
3)              Main Input shear directory: The directory that will houses 
                all of the shear flow simulation data.
4)              Main Output directory: The directory that will house all of the
                output files associated with the analysis; if it doesn't exist,
                it will be created.

CREATED:        26Apr23

MODIFICATIONS
LOG:            

18Jul23         Added new method to main class to plot instances of the filament
                shape for each movement phase. Violinplots can now be omitted
                in favor of swarmplots. 
    
            
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
                poiseuille_shear_data_comp() class itself; it will import the plotting 
                function everytime the plotting routine is called.
2)              Running the method to perform the 2-way ANOVA, ad hoc Tukey test,
                generate the violin plots, and annotate them requires the 
                'statsannotations' library. The current version of 'statsannotations'
                library (v0.5) requires 'seaborn' (<v0.12). If running this script
                through conda, a separate environment is required.

"""

import re, sys, os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})
      
#%%

class poiseuille_shear_data_comp():
    def __init__(self,input_poiseuille_dirs,input_shear_dir,output_dir):
        """
        Intialize the class & locate the directories with the non-Bronwian
        Shear and Poiseuille flow simulation data .
        """
        self.input_poi_dirs = input_poiseuille_dirs
        self.input_shr_dir = input_shear_dir
        self.output_dir = output_dir
    
    def find_corr_dir(self):
        """
        This method finds the corresponding non-Brownian Shear flow simulation
        data for each Poiseuille flow simulation data (both up and down movements).
        """
        self.mubar_shear = lambda mubar_poi,U_c,y_0,channel_h: mubar_poi*(2*U_c*y_0/channel_h**2)
        self.all_corr_dir_list = []
        self.up_shear_data_dirn = 'Up_0p30'
        self.down_shear_data_dirn = 'Down_0p30'
        self.up_poi_dirs = []
        self.down_poi_dirs = []
        
        ### Sort Up vs. Down simulation Data ###
        for dir_ in self.input_poi_dirs:
            for root,dirs,files in os.walk(dir_):
                for subdir in dirs:
                    self.new_path = os.path.join(dir_,root,subdir)
                    if 'Up_' in self.new_path:
                        self.up_poi_dirs.append(self.new_path)
                    elif 'Down_' in self.new_path:
                        self.down_poi_dirs.append(self.new_path)
          
        ### Find the corresponding shear flow simulation data ###
        for self.dir_ in self.up_poi_dirs:
            self.down_dir = self.dir_.replace('Up','Down')
            if self.down_dir in self.down_poi_dirs:
                
                param_csv = pd.read_csv(os.path.join(self.dir_,'parameter_values.csv'),
                                        index_col = 0,header = 0)
                mu_bar = float(param_csv.loc['Mu_bar','Value'])
                U_c = float(param_csv.loc['Poiseuille U Centerline','Value'])
                y_0 = float(param_csv.loc['Vertical Displacement','Value'])
                channel_h = np.abs(float(param_csv.loc['Channel Upper Height','Value']))
                
                self.mubar_shear_eff = self.mubar_shear(mubar_poi = mu_bar,U_c = U_c,
                                                    y_0 = y_0,channel_h = channel_h)
                self.pref_mu_bar = str(np.round((self.mubar_shear_eff/1e5),2)).split('.')
                if len(self.pref_mu_bar[1]) == 1: #If remainder is multiple of 10
                    self.pref_mu_bar[1] = '{}0'.format(self.pref_mu_bar[1])
                self.new_str = 'p'.join(self.pref_mu_bar)
                self.shear_adj_mu_bar_txt = 'MB_{}e5'.format(self.new_str)
                
                self.up_shear_dir = os.path.join(self.input_shr_dir,
                                                  self.shear_adj_mu_bar_txt,
                                                  self.up_shear_data_dirn)
                self.down_shear_dir = os.path.join(self.input_shr_dir,
                                                        self.shear_adj_mu_bar_txt,
                                                        self.down_shear_data_dirn)
                if os.path.exists(self.up_shear_dir) and os.path.exists(self.down_shear_dir):
                    #Append all corresponding directories to a list
                    corr_dir = [self.dir_,self.down_dir,
                                self.up_shear_dir,self.down_shear_dir]
                    self.all_corr_dir_list.append(corr_dir)
        
        
    def append_all_data(self):
        """
        This method calculates all of the center of mass data and appends it
        to a Pandas Dataframe.
        """
        self.com_data_all_lst = []
        
        for self.poi_shr_equiv in self.all_corr_dir_list:
            up_poi_dir,down_poi_dir,up_shr_dir,down_shr_dir = self.poi_shr_equiv
            
            ### Poiseuille Data ###
            # Position Data 
            self.up_poi_position_data = np.load(os.path.join(
                    up_poi_dir,'filament_allstate.npy'))
            self.down_poi_position_data = np.load(os.path.join(
                down_poi_dir,'filament_allstate.npy'))
            
            # Time Data
            self.up_poi_time_data = np.load(os.path.join(
                up_poi_dir,'filament_time_vals_all.npy'))
            self.down_poi_time_data = np.load(os.path.join(
                down_poi_dir,'filament_time_vals_all.npy'))
            
            # Tension Data
            self.up_poi_tension_data = np.load(os.path.join(
                up_poi_dir,'filament_tension.npy'))
            self.down_poi_tension_data = np.load(os.path.join(
                down_poi_dir,'filament_tension.npy'))
            
            # Parameter Values 
            self.up_poi_param_df = pd.read_csv(os.path.join(up_poi_dir,'parameter_values.csv'),
                                          index_col = 0,header = 0)
            self.down_poi_param_df = pd.read_csv(os.path.join(down_poi_dir,'parameter_values.csv'),
                                          index_col = 0,header = 0)
            self.up_poi_max_time_val = float(self.up_poi_param_df.loc['Simulation End Time','Value'])
            self.down_poi_max_time_val = float(self.down_poi_param_df.loc['Simulation End Time','Value'])
            self.up_poi_vd = float(self.up_poi_param_df.loc['Vertical Displacement','Value'])
            self.down_poi_vd = float(self.down_poi_param_df.loc['Vertical Displacement','Value'])
            self.up_poi_mu_bar = float(self.up_poi_param_df.loc['Mu_bar','Value'])
            self.down_poi_mu_bar = float(self.down_poi_param_df.loc['Mu_bar','Value'])
            self.up_poi_channel_h = float(self.up_poi_param_df.loc['Channel Upper Height','Value'])
            self.down_poi_channel_h = float(self.down_poi_param_df.loc['Channel Upper Height','Value'])
            self.up_poi_u_centerline = float(self.up_poi_param_df.loc['Poiseuille U Centerline','Value'])
            self.down_poi_u_centerline = float(self.down_poi_param_df.loc['Poiseuille U Centerline','Value'])
            
            self.up_poi_N = int(self.up_poi_param_df.loc['N','Value'])
            self.up_poi_s = np.linspace(
                float(self.up_poi_param_df.loc['Filament s start','Value']),
                float(self.up_poi_param_df.loc['Filament s end','Value']),
                self.up_poi_N)
            self.up_poi_ds = 1/(self.up_poi_N-1)
            
            self.down_poi_N = int(self.down_poi_param_df.loc['N','Value'])
            self.down_poi_ds = 1/(self.down_poi_N-1)
            self.down_poi_s = np.linspace(
                float(self.down_poi_param_df.loc['Filament s start','Value']),
                float(self.down_poi_param_df.loc['Filament s end','Value']),
                self.down_poi_N)
            
            
            if self.up_poi_channel_h == 0.5 and self.up_poi_u_centerline == 1:
                self.up_poi_confined_type = 'C'
            elif self.up_poi_channel_h == 0.75 and self.up_poi_u_centerline == 2.25:
                self.up_poi_confined_type = 'LC'
            
            if self.up_poi_mu_bar == 5e4:
                self.up_poi_flow_strength = 'W'
            elif self.up_poi_mu_bar == 1e5:
                self.up_poi_flow_strength = 'M'
            if self.up_poi_mu_bar == 2e5:
                self.up_poi_flow_strength = 'S'
                
            if self.down_poi_channel_h == 0.5 and self.down_poi_u_centerline == 1:
                self.down_poi_confined_type = 'C'
            elif self.down_poi_channel_h == 0.75 and self.down_poi_u_centerline == 2.25:
                self.down_poi_confined_type = 'LC'
            
            if self.down_poi_mu_bar == 5e4:
                self.down_poi_flow_strength = 'W'
            elif self.down_poi_mu_bar == 1e5:
                self.down_poi_flow_strength = 'M'
            if self.down_poi_mu_bar == 2e5:
                self.down_poi_flow_strength = 'S'
                
            self.up_poi_flow_type = 'Poiseuille ({}-{})'.format(self.up_poi_flow_strength,self.up_poi_confined_type)
            self.down_poi_flow_type = 'Poiseuille ({}-{})'.format(self.down_poi_flow_strength,self.down_poi_confined_type)
            
            # Find when tension is first negative
            self.up_poi_start_idx = np.where((self.up_poi_tension_data <= 0).all(axis = 0))[0][0]
            self.down_poi_start_idx = np.where((self.down_poi_tension_data <= 0).all(axis = 0))[0][0]
            
            
            # Filter data to be where tension is first negative 
            self.up_poi_position_data = self.up_poi_position_data[:,:,self.up_poi_start_idx:]
            self.down_poi_position_data = self.down_poi_position_data[:,:,self.down_poi_start_idx:]
            
            self.up_poi_time_data = self.up_poi_time_data[self.up_poi_start_idx:]
            self.down_poi_time_data = self.down_poi_time_data[self.down_poi_start_idx:]
            
            
            # Normalize Time Data
            self.up_poi_time_data_adj = self.up_poi_time_data/self.up_poi_max_time_val
            self.down_poi_time_data_adj = self.down_poi_time_data/self.down_poi_max_time_val
            
            # Find center of mass data
            self.up_poi_position_adj = adjust_position_data(position_data = self.up_poi_position_data,
                                                            adj_centering = True,
                                                            adj_translation = True,
                                                            transl_val = self.up_poi_vd)
            
            self.down_poi_position_adj = adjust_position_data(position_data = self.down_poi_position_data,
                                                            adj_centering = True,
                                                            adj_translation = True,
                                                            transl_val = self.down_poi_vd)
            
            self.up_poi_com_vals = center_of_mass(
                position_data = self.up_poi_position_adj,
                position = 0, dim = 3, adj_centering = False, adj_translation = False,
                transl_val = 0)
            self.down_poi_com_vals = center_of_mass(
                position_data = self.down_poi_position_adj,
                position = 0, dim = 3, adj_centering = False, adj_translation = False,
                transl_val = 0)

            # Find unit tangent vector 
            self.up_poi_xs = first_derivative(base_array = self.up_poi_position_data,
                                              deriv_size = self.up_poi_ds, axis = 0, ar_size = self.up_poi_N,
                                              dim = 3)
            
            self.down_poi_xs = first_derivative(base_array = self.down_poi_position_data,
                                              deriv_size = self.down_poi_ds, axis = 0, ar_size = self.up_poi_N,
                                              dim = 3)
            
            
            # Find time values and center of mass change for each stage
            self.up_poi_fil_trans_vals_all = self.find_transition_points(time_data = self.up_poi_time_data.copy(),
                                                                         norm_time_data = self.up_poi_time_data_adj.copy(),
                                                                         center_of_mass_y = self.up_poi_com_vals[:,1].copy(),
                                                                         xs_data =  np.abs(self.up_poi_xs[:,1,:]).max(axis = 0).copy(),
                                                                         xs_thres = 0.99)
            self.up_poi_s1_t,self.up_poi_s2_t,self.up_poi_s3_t,\
                self.up_poi_s1_tn,self.up_poi_s2_tn,self.up_poi_s3_tn,\
            self.up_poi_s1_dy,self.up_poi_s2_dy,self.up_poi_s3_dy = self.up_poi_fil_trans_vals_all[:-2]
            self.up_poi_s1_s2_idx,self.up_poi_s2_s3_idx = self.up_poi_fil_trans_vals_all[-2:]
            
            self.down_poi_fil_trans_vals_all = self.find_transition_points(time_data = self.down_poi_time_data.copy(),
                                                                           norm_time_data = self.down_poi_time_data_adj.copy(),
                                                                           center_of_mass_y = self.down_poi_com_vals[:,1].copy(),
                                                                           xs_data =  np.abs(self.down_poi_xs[:,1,:]).max(axis = 0).copy(),
                                                                           xs_thres = 0.99)
            self.down_poi_s1_t,self.down_poi_s2_t,self.down_poi_s3_t,\
                self.down_poi_s1_tn,self.down_poi_s2_tn,self.down_poi_s3_tn,\
                self.down_poi_s1_dy,self.down_poi_s2_dy,self.down_poi_s3_dy = self.down_poi_fil_trans_vals_all[:-2]
            self.down_poi_s1_s2_idx,self.down_poi_s2_s3_idx = self.down_poi_fil_trans_vals_all[-2:]
            
            
            ### Shear Data ###
            
            # Position Data 
            self.up_shear_position_data = np.load(os.path.join(
                up_shr_dir,'filament_allstate.npy'))
            self.down_shear_position_data = np.load(os.path.join(
                down_shr_dir,'filament_allstate.npy'))
            
            # Time Data
            self.up_shear_time_data = np.load(os.path.join(
                up_shr_dir,'filament_time_vals_all.npy'))
            self.down_shear_time_data = np.load(os.path.join(
                down_shr_dir,'filament_time_vals_all.npy'))
            
            # Tension Data
            self.up_shear_tension_data = np.load(os.path.join(
                up_shr_dir,'filament_tension.npy'))
            self.down_shear_tension_data = np.load(os.path.join(
                up_shr_dir,'filament_tension.npy'))
            
            # Parameter Values 
            self.up_shear_param_df = pd.read_csv(os.path.join(
                up_shr_dir,'parameter_values.csv'),
                                          index_col = 0,header = 0)
            self.down_shear_param_df = pd.read_csv(os.path.join(
                down_shr_dir,'parameter_values.csv'),
                                          index_col = 0,header = 0)
            self.up_shear_max_time_val = float(self.up_shear_param_df.loc['Simulation End Time','Value'])
            self.down_shear_max_time_val = float(self.down_shear_param_df.loc['Simulation End Time','Value'])
            self.up_shear_vd = float(self.up_shear_param_df.loc['Vertical Displacement','Value'])
            self.down_shear_vd = float(self.down_shear_param_df.loc['Vertical Displacement','Value'])
            self.up_shear_mu_bar = float(self.up_shear_param_df.loc['Mu_bar','Value'])
            self.down_shear_mu_bar = float(self.down_shear_param_df.loc['Mu_bar','Value'])
            self.up_shear_channel_h = float(self.up_shear_param_df.loc['Channel Upper Height','Value'])
            self.down_shear_channel_h = float(self.down_shear_param_df.loc['Channel Upper Height','Value'])
            self.up_shear_u_centerline = float(self.up_shear_param_df.loc['Poiseuille U Centerline','Value'])
            self.down_shear_u_centerline = float(self.down_shear_param_df.loc['Poiseuille U Centerline','Value'])
            self.up_shear_flow_type = 'Shear'
            self.down_shear_flow_type = 'Shear'
            
            self.up_shear_N = int(self.up_shear_param_df.loc['N','Value'])
            self.up_shear_s = np.linspace(
                float(self.up_shear_param_df.loc['Filament s start','Value']),
                float(self.up_shear_param_df.loc['Filament s end','Value']),
                self.up_shear_N)
            self.up_shear_ds = 1/(self.up_shear_N-1)
            
            self.down_shear_N = int(self.down_shear_param_df.loc['N','Value'])
            self.down_shear_ds = 1/(self.down_shear_N-1)
            self.down_shear_s = np.linspace(
                float(self.down_shear_param_df.loc['Filament s start','Value']),
                float(self.down_shear_param_df.loc['Filament s end','Value']),
                self.down_shear_N)
            
            # Find when tension is first negative
            self.up_shear_start_idx = np.where((self.up_shear_tension_data <= 0).all(axis = 0))[0][0]
            self.down_shear_start_idx = np.where((self.down_shear_tension_data <= 0).all(axis = 0))[0][0]
            
            # Filter data to be where tension is first negative 
            self.up_shear_position_data = self.up_shear_position_data[:,:,self.up_shear_start_idx:]
            self.down_shear_position_data = self.down_shear_position_data[:,:,self.down_shear_start_idx:]
            
            self.up_shear_time_data = self.up_shear_time_data[self.up_shear_start_idx:]
            self.down_shear_time_data = self.down_shear_time_data[self.down_shear_start_idx:]
            
            # Normalize Time Data
            self.up_shear_time_data_adj = self.up_shear_time_data/self.up_shear_max_time_val
            self.down_shear_time_data_adj = self.down_shear_time_data/self.down_shear_max_time_val
            
            # Find center of mass data
            self.up_shear_position_adj = adjust_position_data(position_data = self.up_shear_position_data,
                                                            adj_centering = True,
                                                            adj_translation = True,
                                                            transl_val = self.up_shear_vd)
            
            self.down_shear_position_adj = adjust_position_data(position_data = self.down_shear_position_data,
                                                            adj_centering = True,
                                                            adj_translation = True,
                                                            transl_val = self.down_shear_vd)
            
            
            # Find center of mass data
            self.up_shear_com_vals = center_of_mass(
                position_data = self.up_shear_position_adj,
                position = 0, dim = 3, adj_centering = False, adj_translation = False,
                transl_val = 0)
            self.down_shear_com_vals = center_of_mass(
                position_data = self.down_shear_position_adj,
                position = 0, dim = 3, adj_centering = False, adj_translation = False,
                transl_val = 0)
            
            # Find unit tangent vector 
            self.up_shear_xs = first_derivative(base_array = self.up_shear_position_data,
                                              deriv_size = self.up_shear_ds, axis = 0, ar_size = self.up_shear_N,
                                              dim = 3)
            
            self.down_shear_xs = first_derivative(base_array = self.down_shear_position_data,
                                              deriv_size = self.down_shear_ds, axis = 0, ar_size = self.up_shear_N,
                                              dim = 3)
            
            # Find time values and center of mass change for each stage
            self.up_shear_fil_trans_vals_all = self.find_transition_points(time_data = self.up_shear_time_data.copy(),
                                                                           norm_time_data = self.up_shear_time_data_adj.copy(),
                                                                           center_of_mass_y = self.up_shear_com_vals[:,1].copy(),
                                                                           xs_data =  np.abs(self.up_shear_xs[:,1,:]).max(axis = 0).copy(),
                                                                           xs_thres = 0.99)
            self.up_shear_s1_t,self.up_shear_s2_t,self.up_shear_s3_t,\
                self.up_shear_s1_tn,self.up_shear_s2_tn,self.up_shear_s3_tn,\
            self.up_shear_s1_dy,self.up_shear_s2_dy,self.up_shear_s3_dy = self.up_shear_fil_trans_vals_all[:-2]
            self.up_shear_s1_s2_idx,self.up_shear_s2_s3_idx = self.up_shear_fil_trans_vals_all[-2:]
            
            self.down_shear_fil_trans_vals_all = self.find_transition_points(time_data = self.down_shear_time_data.copy(),
                                                                             norm_time_data = self.down_shear_time_data_adj.copy(),
                                                                             center_of_mass_y = self.down_shear_com_vals[:,1].copy(),
                                                                             xs_data =  np.abs(self.down_shear_xs[:,1,:]).max(axis = 0).copy(),
                                                                             xs_thres = 0.99)
            self.down_shear_s1_t,self.down_shear_s2_t,self.down_shear_s3_t,\
                self.down_shear_s1_tn,self.down_shear_s2_tn,self.down_shear_s3_tn,\
                self.down_shear_s1_dy,self.down_shear_s2_dy,self.down_shear_s3_dy = self.down_shear_fil_trans_vals_all[:-2]
            self.down_shear_s1_s2_idx,self.down_shear_s2_s3_idx = self.down_shear_fil_trans_vals_all[-2:]
            
            
            ### Save Results into List that gets converted into DataFrame ###
            self.poi_up_data = {"Flow Type": self.up_poi_flow_type,
                        "Mu_bar": self.up_poi_mu_bar,
                        "Displacement Type": 'Up',
                        "Starting Vertical Displacement": self.up_poi_vd,
                        "Channel Height": self.up_poi_channel_h,
                        "Poiseuille U Centerline": self.up_poi_u_centerline,
                        "ABS Net Adjusted Center of Mass-y": self.up_poi_com_vals[:,1],
                        "Time": self.up_poi_time_data,
                        "Normalized Time": self.up_poi_time_data_adj,
                        "Stage 1 ABS Net Adjusted Center of Mass-y": self.up_poi_s1_dy,
                        "Stage 2 ABS Net Adjusted Center of Mass-y": self.up_poi_s2_dy,
                        "Stage 3 ABS Net Adjusted Center of Mass-y": self.up_poi_s3_dy,
                        "Stage 1 Time": self.up_poi_s1_t,
                        "Stage 2 Time": self.up_poi_s2_t,
                        "Stage 3 Time": self.up_poi_s3_t,
                        "Stage 1 Normalized Time": self.up_poi_s1_tn,
                        "Stage 2 Normalized Time": self.up_poi_s2_tn,
                        "Stage 3 Normalized Time": self.up_poi_s3_tn,
                        "Stage 1-2 Time Index":self.up_poi_s1_s2_idx,
                        "Stage 2-3 Time Index":self.up_poi_s2_s3_idx}
            
            self.poi_down_data = {"Flow Type": self.down_poi_flow_type,
                        "Mu_bar": self.down_poi_mu_bar,
                        "Displacement Type": 'Down',
                        "Starting Vertical Displacement": self.down_poi_vd,
                        "Channel Height": self.down_poi_channel_h,
                        "Poiseuille U Centerline": self.down_poi_u_centerline,
                        "ABS Net Adjusted Center of Mass-y": np.abs(self.down_poi_com_vals[:,1]),
                        "Time": self.down_poi_time_data,
                        "Normalized Time": self.down_poi_time_data_adj,
                        "Stage 1 ABS Net Adjusted Center of Mass-y": self.down_poi_s1_dy,
                        "Stage 2 ABS Net Adjusted Center of Mass-y": self.down_poi_s2_dy,
                        "Stage 3 ABS Net Adjusted Center of Mass-y": self.down_poi_s3_dy,
                        "Stage 1 Time": self.down_poi_s1_t,
                        "Stage 2 Time": self.down_poi_s2_t,
                        "Stage 3 Time": self.down_poi_s3_t,
                        "Stage 1 Normalized Time": self.down_poi_s1_tn,
                        "Stage 2 Normalized Time": self.down_poi_s2_tn,
                        "Stage 3 Normalized Time": self.down_poi_s3_tn,
                        "Stage 1-2 Time Index":self.down_poi_s1_s2_idx,
                        "Stage 2-3 Time Index":self.down_poi_s2_s3_idx}
            
            self.shr_up_data = {"Flow Type": self.up_shear_flow_type,
                        "Mu_bar": self.up_shear_mu_bar,
                        "Displacement Type": 'Up',
                        "Starting Vertical Displacement": self.up_shear_vd,
                        "Channel Height": self.up_shear_channel_h,
                        "Poiseuille U Centerline": self.down_shear_u_centerline,
                        "ABS Net Adjusted Center of Mass-y": np.abs(self.up_shear_com_vals[:,1]),
                        "Time": self.up_shear_time_data,
                        "Normalized Time": self.up_shear_time_data_adj,
                        "Stage 1 ABS Net Adjusted Center of Mass-y": self.up_shear_s1_dy,
                        "Stage 2 ABS Net Adjusted Center of Mass-y": self.up_shear_s2_dy,
                        "Stage 3 ABS Net Adjusted Center of Mass-y": self.up_shear_s3_dy,
                        "Stage 1 Time": self.up_shear_s1_t,
                        "Stage 2 Time": self.up_shear_s2_t,
                        "Stage 3 Time": self.up_shear_s3_t,
                        "Stage 1 Normalized Time": self.up_shear_s1_tn,
                        "Stage 2 Normalized Time": self.up_shear_s2_tn,
                        "Stage 3 Normalized Time": self.up_shear_s3_tn,
                        "Stage 1-2 Time Index":self.up_shear_s1_s2_idx,
                        "Stage 2-3 Time Index":self.up_shear_s2_s3_idx}
            
            self.shr_down_data = {"Flow Type": self.down_shear_flow_type,
                        "Mu_bar": self.down_shear_mu_bar,
                        "Displacement Type": 'Down',
                        "Starting Vertical Displacement": self.down_shear_vd,
                        "Channel Height": self.down_shear_channel_h,
                        "Poiseuille U Centerline": self.down_shear_u_centerline,
                        "ABS Net Adjusted Center of Mass-y": np.abs(self.down_shear_com_vals[:,1]),
                        "Time": self.down_shear_time_data,
                        "Normalized Time": self.down_shear_time_data_adj,
                        "Stage 1 ABS Net Adjusted Center of Mass-y": self.down_shear_s1_dy,
                        "Stage 2 ABS Net Adjusted Center of Mass-y": self.down_shear_s2_dy,
                        "Stage 3 ABS Net Adjusted Center of Mass-y": self.down_shear_s3_dy,
                        "Stage 1 Time": self.down_shear_s1_t,
                        "Stage 2 Time": self.down_shear_s2_t,
                        "Stage 3 Time": self.down_shear_s3_t,
                        "Stage 1 Normalized Time": self.down_shear_s1_tn,
                        "Stage 2 Normalized Time": self.down_shear_s2_tn,
                        "Stage 3 Normalized Time": self.down_shear_s3_tn,
                        "Stage 1-2 Time Index":self.down_shear_s1_s2_idx,
                        "Stage 2-3 Time Index":self.down_shear_s2_s3_idx}
            
            self.com_data_all_lst.extend([self.poi_up_data,self.poi_down_data,
                                         self.shr_up_data,self.shr_down_data])
            
        self.all_poi_shr_data_df = pd.concat(
            [pd.DataFrame.from_dict(i) for i in self.com_data_all_lst],
            ignore_index = True)
        self.all_poi_shr_data_df['Flow Type-Displacement Type'] = self.all_poi_shr_data_df['Flow Type'] + '-' +\
            self.all_poi_shr_data_df['Displacement Type']
            

    
    
    def find_transition_points(self,time_data,norm_time_data,center_of_mass_y,xs_data,xs_thres):
        """
        This method finds the transition points of the center of mass curve
        under the assumption that the second stage (snake-like motion) has a 
        maximum value of the y-component of the unit tangent vector x_s of 
        approximately 1. 
        
        """
        
        self.j_loop_idx = np.where(xs_data.copy() >= xs_thres)[0]
        self.j_loop_start,self.j_loop_end = self.j_loop_idx[0],self.j_loop_idx[-1]
        
        self.stage_1_time = time_data[self.j_loop_start] - time_data[0]
        self.stage_2_time = time_data[self.j_loop_end] - time_data[self.j_loop_start]
        self.stage_3_time = time_data[-1] - time_data[self.j_loop_end]
        
        self.stage_1_time_norm = norm_time_data[self.j_loop_start] - norm_time_data[0]
        self.stage_2_time_norm = norm_time_data[self.j_loop_end] - norm_time_data[self.j_loop_start]
        self.stage_3_time_norm = norm_time_data[-1] - norm_time_data[self.j_loop_end]
        
        self.stage_1_net_y = np.abs(center_of_mass_y[self.j_loop_start] - center_of_mass_y[0])
        self.stage_2_net_y = np.abs(center_of_mass_y[self.j_loop_end] - center_of_mass_y[self.j_loop_start])
        self.stage_3_net_y = np.abs(center_of_mass_y[-1] - center_of_mass_y[self.j_loop_end])
        
        return [self.stage_1_time,self.stage_2_time,self.stage_3_time,\
                self.stage_1_time_norm,self.stage_2_time_norm,self.stage_3_time_norm,\
                    self.stage_1_net_y,self.stage_2_net_y,self.stage_3_net_y,self.j_loop_start,self.j_loop_end]
    
    
    def calculate_rescale_values(self):
        """
        This method calculates the effective shear flow mu_bar for all data.
        """

        
        """Select Poiseuille (based on Channel Height & Mu_bar) vs. Shear"""
        
        fil_all_poi_data_df = self.all_poi_shr_data_df[
            (self.all_poi_shr_data_df['Flow Type'] != 'Shear')].copy()
        fil_all_shr_data_df = self.all_poi_shr_data_df[
            (self.all_poi_shr_data_df['Flow Type'] == 'Shear')].copy()
        all_poi_data_idx_vals = fil_all_poi_data_df.index.values
        all_shr_data_idx_vals = fil_all_shr_data_df.index.values
        
        ### Calculate effective Shear Flow Mu_bar ###
        self.all_poi_shr_data_df.loc[all_poi_data_idx_vals,'Effective Shear Mu_bar'] = fil_all_poi_data_df['Mu_bar']*\
            2*fil_all_poi_data_df['Poiseuille U Centerline']*fil_all_poi_data_df['Starting Vertical Displacement']/\
                fil_all_poi_data_df['Channel Height']**2
        self.all_poi_shr_data_df.loc[all_shr_data_idx_vals,'Effective Shear Mu_bar'] = fil_all_shr_data_df['Mu_bar']
    
        ### Calculate Re-scaled Mu_bar (mu_bar_eff/mu_bar_poi) ###
        self.all_poi_shr_data_df.loc[all_poi_data_idx_vals,'Rescaled Mu_bar'] = self.all_poi_shr_data_df.loc[all_poi_data_idx_vals,'Effective Shear Mu_bar']/\
            self.all_poi_shr_data_df.loc[all_poi_data_idx_vals,'Mu_bar']
        self.all_poi_shr_data_df.loc[all_shr_data_idx_vals,'Rescaled Mu_bar'] = self.all_poi_shr_data_df.loc[all_shr_data_idx_vals,'Effective Shear Mu_bar']
        
        ### Rescale Vertical Displacement (Delta y * mu_bar_poi^(1/8) ###
        # self.all_poi_shr_data_df.loc[all_poi_data_idx_vals,'Rescaled ABS Adjusted Center of Mass-y'] = self.all_poi_shr_data_df.loc[all_poi_data_idx_vals,'ABS Adjusted Center of Mass-y']*\
        #     (self.all_poi_shr_data_df.loc[all_poi_data_idx_vals,'Mu_bar'])**(.125)
        self.all_poi_shr_data_df.loc[all_poi_data_idx_vals,'Rescaled ABS Net Adjusted Center of Mass-y'] = self.all_poi_shr_data_df.loc[all_poi_data_idx_vals,'ABS Net Adjusted Center of Mass-y']*\
            (self.all_poi_shr_data_df.loc[all_poi_data_idx_vals,'Effective Shear Mu_bar'])**(.125)
        self.all_poi_shr_data_df.loc[all_shr_data_idx_vals,'Rescaled ABS Net Adjusted Center of Mass-y'] = self.all_poi_shr_data_df.loc[all_shr_data_idx_vals,'ABS Net Adjusted Center of Mass-y']
   
            
    def calc_shear_ycom_displ_regression(self):
        """
        This method calculates the scaling factor between the adjusted net 
        center of mass displacement vs. effective shear flow mu_bar for shear
        flow data.
        """
        fil_shear_data_df = self.all_poi_shr_data_df.copy()
        fil_shear_data_df = fil_shear_data_df[(fil_shear_data_df['Flow Type'] == 'Shear') &\
                                              (fil_shear_data_df['Normalized Time'] == 1)]
        fil_shear_data_df['Log10 ABS Net Adjusted Center of Mass-y'] = np.log10(fil_shear_data_df['ABS Net Adjusted Center of Mass-y'])
        fil_shear_data_df['Log10 Effective Shear Mu_bar'] = np.log10(fil_shear_data_df['Effective Shear Mu_bar'])
        self.model = smf.ols('Q("Log10 ABS Net Adjusted Center of Mass-y") ~ Q("Log10 Effective Shear Mu_bar")',data = fil_shear_data_df).fit()
        self.intercept,self.slope,self.rsquared = 10**self.model.params[0],self.model.params[1],self.model.rsquared
    
    def fit_function(self,x,a):
        """
        This method is used to fit the net displacement of the shear flow data to 
        the following scaling relationship:
            
        Delta_y = a*mu_bar^(-0.125)
        
        Where a is a constant.
        
        Inputs:
            
        x:      Mu_bar values.
        a:      Constant value to be solved.
        """
        return a*x**(-0.125)
    
    def do_curve_fit(self):
        """
        This method performs a curve fitting to Net drift ~ Mu_bar ^(-1/8).
        """
        fil_shear_data_df = self.all_poi_shr_data_df.copy()
        fil_shear_data_df = fil_shear_data_df[(fil_shear_data_df['Flow Type'] == 'Shear') &\
                                              (fil_shear_data_df['Normalized Time'] == 1)]
        self.intercept_2 = curve_fit(self.fit_function,xdata = fil_shear_data_df['Effective Shear Mu_bar'].to_numpy(),
                                     ydata = fil_shear_data_df['ABS Net Adjusted Center of Mass-y'].to_numpy())[0][0]
        
    def sort_stage_data(self):
        """
        This method sorts the Net Y-com stage data into a DataFrame that can
        be used to plot the violin plots.
        """
        self.cp_data_df = self.all_poi_shr_data_df.copy()
        self.cp_data_df = self.cp_data_df[(self.cp_data_df['Flow Type'] != 'Shear') &\
                        (self.cp_data_df['Poiseuille U Centerline'] == 1) &\
                            (self.cp_data_df['Mu_bar'] == 1e5) & (self.cp_data_df['Normalized Time'] == 1)]
        self.cols_to_expand = ['Stage 1 ABS Net Adjusted Center of Mass-y','Stage 2 ABS Net Adjusted Center of Mass-y','Stage 3 ABS Net Adjusted Center of Mass-y']
        self.cols_to_discard = ['Time','Normalized Time']
        self.cols_to_keep = [i for i in self.cp_data_df.columns if i not in self.cols_to_expand and i not in self.cols_to_discard]
        self.exp_cp_data_df = pd.melt(self.cp_data_df,id_vars = self.cols_to_keep,value_vars = self.cols_to_expand,
                          var_name = 'Stage',value_name = 'Stage ABS Adjusted Center of Mass-y')
        self.exp_cp_data_df.drop_duplicates(inplace = True)
        self.exp_cp_data_df.replace({"Stage 1 ABS Net Adjusted Center of Mass-y":"I",
                             "Stage 2 ABS Net Adjusted Center of Mass-y": "II",
                             "Stage 3 ABS Net Adjusted Center of Mass-y": "III"},inplace = True)
        
        
    def create_dir(self,dir_):
        """
        This method creates the output directory if it doesn't already exist.
        """
        create_dir(dir_)
        
        
    def plot_net_com_comp(self,vert_displ):
        """
        This method plots the adjusted net center of mass displacement as a 
        function of the effective shear flow mu_bar for both Poiseulle and 
        Shear flow data on a scatter plot.
        """
        from plotting.net_ycom_comp_mu_bar_scaling import net_ycom_comp_mu_bar_scaling as drift_scaling
        new_dir = os.path.join(self.output_dir,'all_data/')
        filtered_df = self.all_poi_shr_data_df.copy()
        filtered_df = filtered_df[(filtered_df['Flow Type'] == 'Shear') |\
                                  (filtered_df['Poiseuille U Centerline'] == 1.00)]
        filtered_df.replace({"Poiseuille (W-C)":"Poiseuille (W)",
                             "Poiseuille (M-C)": "Poiseuille (M)",
                             "Poiseuille (S-C)": "Poiseuille (S)"},inplace = True)
        self.create_dir(new_dir)
        drift_scaling(data_df = filtered_df,
                          fit_slope = -0.125,
                          fit_intercept = self.intercept_2,
                          output_dir = new_dir)
        

        
    def plot_ycom_time_curves_flow_comp(self,vert_displ,poi_flow_type,poi_mu_bar,poi_Uc):
        """
        This method plots Delta y as a function of time for both an upward & downward
        flip in Poiseuille flow alongside the same flip types for Shear flow. 
        The mu_bar for shear flow is calculated to be the effective shear mu_bar 
        based on the fact that mu_bar times the velocity gradient needs to be 
        equivalent to each other between shear and poiseuille flow.
        """
        from plotting.ycom_time_curves_flow_comp import com_time_curves_flow_comp as ycom_time_flow_comp
        new_dir = os.path.join(self.output_dir,'shear_poi_com_curves/')
        self.create_dir(new_dir)
        ycom_time_flow_comp(data_df = self.all_poi_shr_data_df, 
                              output_dir = new_dir,
                              position_vals = vert_displ,
                              poi_flow_type = poi_flow_type,
                              poi_mu_bar = poi_mu_bar,poi_Uc = poi_Uc)
        
        
    def ycom_time_curves_poi_all(self,poi_flow_type,poi_mu_bar,poi_Uc):
        """
        This method plots Delta y as a function of time for both an upward & downward
        flip in Poiseuille flow. The initial condition is denoted with a color
        gradient and color bar.
        """
        from plotting.ycom_time_curves_poi_all import ycom_time_curves_poi_all
        new_dir = os.path.join(self.output_dir,'shear_poi_com_curves_same/')
        self.create_dir(new_dir)
        ycom_time_curves_poi_all(data_df = self.all_poi_shr_data_df, 
                              output_dir = new_dir,
                              poi_flow_type = poi_flow_type,
                              poi_mu_bar = poi_mu_bar,poi_Uc = poi_Uc)
        
        
    def plot_net_ycom_stage_distr(self):
        """
        This method performs a 2-way ANOVA to assess whether or not the net drift
        between upward and downward flips in each of the different Phases in 
        Poiseuille flow. Afterwards it will plot the results as a violinplot with
        the statistics assessed.
        """
        from calculations.ANOVA2_netycom_phase_flipdir import ANOVA2_netycom_phase_flipdir
        from plotting.net_ycom_stage_distr import net_ycom_stage_distr, net_ycom_stage_distr_vert
        new_dir = os.path.join(self.output_dir,'violin_ycom_stage/')
        self.create_dir(new_dir)
        
        self.ycom_stage_pvals = ANOVA2_netycom_phase_flipdir(input_file = self.exp_cp_data_df,
                                                                 output_directory = new_dir,
                                                                 file_name = 'p_vals_ycom_stage')
        
        net_ycom_stage_distr(input_file = self.exp_cp_data_df,p_vals_df = self.ycom_stage_pvals,
                                output_directory = new_dir,file_name = 'violin_ycom_stage')
        
        net_ycom_stage_distr_vert(input_file = self.exp_cp_data_df,p_vals_df = self.ycom_stage_pvals,
                               output_directory = new_dir,file_name = 'violin_ycom_stage_v')
        
        
    def net_ycom_stage_distr_shape_snps(self,up_poi_position_data,down_poi_position_data):
        from plotting.net_ycom_stage_distr_shape_snps import net_ycom_stage_distr_shape_snps
        new_dir = os.path.join(self.output_dir,'violin_ycom_stage_snapshots/')
        self.create_dir(new_dir)

        net_ycom_stage_distr_shape_snps(input_file = self.exp_cp_data_df,
                                               up_poi_position_data = up_poi_position_data,
                                               down_poi_position_data = down_poi_position_data,
                               output_directory = new_dir,file_name = 'SNP_NYCOM_SWRM')
        
        
 
#%% Read Data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_directory",
                        help="Specify the directory where the main script and accompany scripts are located in",
                    type = str)
    parser.add_argument("input_shear_directory",
                        help="Specify the parent directory of the Poiseuille flow Non-Brownian Simulations",
                    type = str)
    parser.add_argument("output_directory",
                        help="Specify the parent directory of the Migration Data",
                    type = str)
    parser.add_argument("--input_poiseuille_directory",'-input_poi_dir',nargs = '+',
                        help="Specify the path to the main directory of the Poiseuille flow Non-Brownian Simulations",
                    type = str)
    parser.add_argument("--input_up_poiseuille_data_directory","-up_poi_dir",nargs = 1,
                         help = "Specify the path to the directory of an upward flip in Poiseuille flow that you want to display snapshots for")
    parser.add_argument("--input_down_poiseuille_data_directory","-down_poi_dir",nargs = 1,
                         help = "Specify the path to the directory of an downward flip in Poiseuille flow that you want to display snapshots for")
    parser.add_argument("--statistical_analysis",'--do_stats',
                        help = "Specify whether or not you want to perform statistical analysis between the net drift of the different movement phases and direction (Requires 'statsannotations' library)",
                        action = 'store_true',default = False)
    
    args = parser.parse_args(['C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01b_Non_Brownian_Analysis\\',
                              'C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01_Migration_Simulations\\02_Actual_Results\\NB_Shear_J\\',
                              'C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01b_Non_Brownian_Analysis\\01_Actual_Results\\Shear_Poi_COM_Up_Down_Comparison\\',
                              '--input_poiseuille_directory',
                              'C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01_Migration_Simulations\\02_Actual_Results\\NB_Poiseuille_J\\',
                              '--input_up_poiseuille_data_directory',
                              'C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01_Migration_Simulations\\02_Actual_Results\\NB_Poiseuille_J\\UC_1p00_MB_1p00e5\\Up_0p30\\',
                              '--input_down_poiseuille_data_directory',
                              'C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01_Migration_Simulations\\02_Actual_Results\\NB_Poiseuille_J\\UC_1p00_MB_1p00e5\\Down_0p30\\'])
    
    # ### comment this line if running from terminal ###
    current_dir = args.project_directory
    os.chdir(current_dir)
    
    from misc.create_dir import create_dir
    from calculations.adjust_position_data import adjust_position_data
    from calculations.center_of_mass import center_of_mass
    from calculations.first_derivative import first_derivative

    data_comp = poiseuille_shear_data_comp(input_poiseuille_dirs= args.input_poiseuille_directory,
                                            input_shear_dir = args.input_shear_directory,
                                            output_dir = args.output_directory)
    
    vert_displ1 = np.arange(0.25,0.46,0.05) #Select ones 
    vert_displ2 = np.array([0.25,0.26,0.28,0.30,0.32,0.34,
                            0.35,0.36,0.38,0.40,0.42,0.44,0.45]) #all of them]
    data_comp.create_dir(data_comp.output_dir)
    data_comp.find_corr_dir()
    data_comp.append_all_data()
    data_comp.calculate_rescale_values()
    data_comp.calc_shear_ycom_displ_regression()
    data_comp.do_curve_fit()
    data_comp.sort_stage_data()
    
    ### Plot ycom vs. effective shear mu_bar
    data_comp.plot_net_com_comp(vert_displ2)
    
    ### Plot com vs. time curves ###
    # data_comp.plot_ycom_time_curves_flow_comp(vert_displ = vert_displ1,
    #                                 poi_flow_type = 'Poiseuille (M-C)',
    #                                 poi_mu_bar = 1e5,poi_Uc = 1)
    data_comp.ycom_time_curves_poi_all(poi_flow_type = 'Poiseuille (M-C)',
                                    poi_mu_bar = 1e5,poi_Uc = 1)
    
    ### Plot swarm/strip plot of delta ycom vs. stage with p-values (need statsannotations)
    # if args.statistical_analysis:
        # data_comp.plot_net_ycom_stage_distr()

    
    ### Plot violin plot of delta ycom without p-values and snapshots of filament 
    if args.input_up_poiseuille_data_directory and args.input_down_poiseuille_data_directory:
        up_poi_poisition_data = np.load(os.path.join(args.input_up_poiseuille_data_directory[0],'filament_allstate.npy'))
        down_poi_poisition_data = np.load(os.path.join(args.input_down_poiseuille_data_directory[0],'filament_allstate.npy'))
        
        adj_up_poi_loc_data = adjust_position_data(position_data = up_poi_poisition_data,
                                                        adj_centering = True,
                                                        adj_translation = True,
                                                        transl_val = 0)
        adj_down_poi_loc_data = adjust_position_data(position_data = down_poi_poisition_data,
                                                        adj_centering = True,
                                                        adj_translation = True,
                                                        transl_val = 0)
        
        data_comp.net_ycom_stage_distr_shape_snps(up_poi_position_data = adj_up_poi_loc_data,
                                          down_poi_position_data = adj_down_poi_loc_data)
