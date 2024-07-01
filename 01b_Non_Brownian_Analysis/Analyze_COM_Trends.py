# -*- coding: utf-8 -*-
"""
FILE NAME:      Analyze_COM_Trends.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        create_dir.py; adjust_position_data.py; center_of_mass.py; true_center.py;
                first_derivative.py; second_derivative.py; plot_filament_com_xs_transitions.py


DESCRIPTION:    This script will read in the filament position data and simulation
                parameters performing an upward/downward flip in either Shear
                or Poiseuille flow. Based on the position data, it will plot the
                absolute value of the center of mass as a function of time.
                It will also calculate the first and second spatial derivatives 
                to approximate the stages of the transition stages of the center
                of mass curve. 

INPUT
FILES(S):       This script will read the following files based on the directories
                of the non-Brownian simulation data in Poiseuille/Shear flow
                performing an upward/downward flip (3 files per directory, 4
                directories total):

1)              .NPY file that contains the filament position data(Nx3xT array)
2)              .NPY file that contains the time datapoints used for the simulation.
3)              .CSV file that contains the filament parameters used for the 
                simulation.

OUTPUT
FILES(S):       

1)              .PDF/.SVG file that shows center of mass curves as a function of
                time for an upward/downward flip in Shear/Poiseuille flow. A curve
                that represents the maximum value of the absolute value of the 
                y-component of the first derivative will be superimposed on
                this graph as well.

INPUT
ARGUMENT(S):    
    
1)              Project Directory: The directory that contains all of the 
                complementary scripts needed to run the analysis.
2)              Main Input Poiseuille directory: The directory that contains the 
                simulation data for an upward and downward flip. Note that the 
                current version of the script will choose a filament initially
                position at y_0 = 0.30.
3)              Main Input Shear directory: The directory that contains the 
                simulation data for an upward and downward flip. Note that the 
                current version of the script will choose a filament initially
                position at y_0 = 0.30.
4)              Main Output directory: The directory that will house all of the
                output files associated with the analysis; if it doesn't exist,
                it will be created.

CREATED:        19Jun23

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

NOTE(S):        
    
1)              The dependency of the he plotting routine is incased in the 
                poi_shear_data() class itself; it will import the plotting 
                function everytime the plotting routine is called.
2)              The resulting plot from this script has transparent components,
                which is supposedly unsupported by .PNG/.TIFF/.EPS files; thus,
                save the plots. as .PDF/.SVG .

"""
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})


class poi_shear_data():
    
    def __init__(self,up_poi_dir,down_poi_dir,up_shear_dir,down_shear_dir,output_dir):
        """
        This method initializes the class and saves the directory containing
        the data files for Poiseuille/Shear flow.
        """
        self.up_poi_dir = up_poi_dir
        self.down_poi_dir = down_poi_dir
        self.up_shr_dir = up_shear_dir
        self.down_shr_dir = down_shear_dir
        self.output_dir = output_dir
        
    def read_poi_data(self):
        """
        This method reads the data files corresponding to Poiseuille flow and
        extracts relevant parameters.
        """

        # Position Data 
        self.up_poi_position_data = np.load(os.path.join(
                self.up_poi_dir,'filament_allstate.npy'))
        self.down_poi_position_data = np.load(os.path.join(
            self.down_poi_dir,'filament_allstate.npy'))
        
        # Time Data
        self.up_poi_time_data = np.load(os.path.join(
            self.up_poi_dir,'filament_time_vals_all.npy'))
        self.down_poi_time_data = np.load(os.path.join(
            self.down_poi_dir,'filament_time_vals_all.npy'))
        
        # Tension Data
        self.up_poi_tension_data = np.load(os.path.join(
            self.up_poi_dir,'filament_tension.npy'))
        self.down_poi_tension_data = np.load(os.path.join(
            self.down_poi_dir,'filament_tension.npy'))
        
        
        # Parameter Values 
        self.up_poi_param_df = pd.read_csv(os.path.join(self.up_poi_dir,'parameter_values.csv'),
                                      index_col = 0,header = 0)
        self.down_poi_param_df = pd.read_csv(os.path.join(self.down_poi_dir,'parameter_values.csv'),
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
        
        self.up_poi_center_idx = np.where(self.up_poi_s == 0)[0][0]
        self.down_poi_center_idx = np.where(self.down_poi_s == 0)[0][0]
        
    
    def read_shear_data(self):
        """
        This method reads the data files corresponding to Shear flow and
        extracts relevant parameters.
        """

        # Position Data 
        self.up_shear_position_data = np.load(os.path.join(
            self.up_shr_dir,'filament_allstate.npy'))
        self.down_shear_position_data = np.load(os.path.join(
            self.down_shr_dir,'filament_allstate.npy'))
        
        # Time Data
        self.up_shear_time_data = np.load(os.path.join(
            self.up_shr_dir,'filament_time_vals_all.npy'))
        self.down_shear_time_data = np.load(os.path.join(
            self.down_shr_dir,'filament_time_vals_all.npy'))
        
        # Tension Data
        self.up_shear_tension_data = np.load(os.path.join(
            self.up_shr_dir,'filament_tension.npy'))
        self.down_shear_tension_data = np.load(os.path.join(
            self.up_shr_dir,'filament_tension.npy'))
        
        # Parameter Values 
        self.up_shear_param_df = pd.read_csv(os.path.join(
            self.up_shr_dir,'parameter_values.csv'),
                                      index_col = 0,header = 0)
        self.down_shear_param_df = pd.read_csv(os.path.join(
            self.down_shr_dir,'parameter_values.csv'),
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
        
        self.up_shear_N = int(self.up_shear_param_df.loc['N','Value'])
        self.up_shear_ds = 1/(self.up_shear_N-1)
        self.down_shear_N = int(self.down_shear_param_df.loc['N','Value'])
        self.down_shear_ds = 1/(self.down_shear_N-1)
        
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
        
        self.up_shear_center_idx = np.where(self.up_shear_s == 0)[0][0]
        self.down_shear_center_idx = np.where(self.down_shear_s == 0)[0][0]
        
        
    def adjust_positions(self):
        """
        This method adjusts the filament position data by keep the x-component
        within frame between -0.5 and 0.5 and subtract out the initial position
        from the y-component if desired.
        """
    
        self.up_poi_position_adj = adjust_position_data(position_data = self.up_poi_position_data,
                                                        adj_centering = True,
                                                        adj_translation = True,
                                                        transl_val = self.up_poi_vd)
        
        self.down_poi_position_adj = adjust_position_data(position_data = self.down_poi_position_data,
                                                        adj_centering = True,
                                                        adj_translation = True,
                                                        transl_val = self.down_poi_vd)
        
        self.up_shear_position_adj = adjust_position_data(position_data = self.up_shear_position_data,
                                                        adj_centering = True,
                                                        adj_translation = True,
                                                        transl_val = self.up_shear_vd)
        
        self.down_shear_position_adj = adjust_position_data(position_data = self.down_shear_position_data,
                                                        adj_centering = True,
                                                        adj_translation = True,
                                                        transl_val = self.down_shear_vd)
        
        
    def calc_center_of_mass(self):
        """
        This method calculates the filament center of mass at every timepoint
        and adjusts for any translation.
        """
        self.up_poi_com_vals = center_of_mass(
            position_data = self.up_poi_position_adj,
            position = 0, dim = 3, adj_centering = False, adj_translation = False,
            transl_val = 0)
        self.down_poi_com_vals = center_of_mass(
            position_data = self.down_poi_position_adj,
            position = 0, dim = 3, adj_centering = False, adj_translation = False,
            transl_val = 0)
        
        self.up_shear_com_vals = center_of_mass(
            position_data = self.up_shear_position_adj,
            position = 0, dim = 3, adj_centering = False, adj_translation = False,
            transl_val = 0)
        self.down_shear_com_vals = center_of_mass(
            position_data = self.down_shear_position_adj,
            position = 0, dim = 3, adj_centering = False, adj_translation = False,
            transl_val = 0)
    
    
    def calc_true_center(self):
        """
        This method finds the true center of the filament at every timepoint.
        """
        self.up_poi_tc_vals = true_center(position_data = self.up_poi_position_adj,
                                          center_idx = self.up_poi_center_idx,
                                          adj_translation = False, transl_val = 0)
        self.down_poi_tc_vals = true_center(position_data = self.down_poi_position_adj,
                                          center_idx = self.down_poi_center_idx,
                                          adj_translation = False, transl_val = 0)
        
        self.up_shear_tc_vals = true_center(position_data = self.up_shear_position_adj,
                                          center_idx = self.up_shear_center_idx,
                                          adj_translation = False, transl_val = 0)
        self.down_shear_tc_vals = true_center(position_data = self.down_shear_position_adj,
                                          center_idx = self.down_shear_center_idx,
                                          adj_translation = False, transl_val = 0)
        
        
    def calc_derivatives(self):
        """
        This method calculates the first and second spatial derivatives of 
        the filament in both Poiseuille and shear flow. 
        """
        self.up_poi_xs = first_derivative(base_array = self.up_poi_position_data,
                                          deriv_size = self.up_poi_ds, axis = 0, 
                                          ar_size = self.up_poi_N, dim = 3)
        self.down_poi_xs = first_derivative(base_array = self.down_poi_position_data,
                                          deriv_size = self.down_poi_ds, axis = 0, 
                                          ar_size = self.down_poi_N, dim = 3)
        self.up_shear_xs = first_derivative(base_array = self.up_shear_position_data,
                                          deriv_size = self.up_shear_ds, axis = 0, 
                                          ar_size = self.up_shear_N, dim = 3)
        self.down_shear_xs = first_derivative(base_array = self.down_shear_position_data,
                                          deriv_size = self.down_shear_ds, axis = 0, 
                                          ar_size = self.down_shear_N, dim = 3)
        
        self.up_poi_xss = second_derivative(base_array = self.up_poi_position_data,
                                          deriv_size = self.up_poi_ds, axis = 0, 
                                          ar_size = self.up_poi_N, dim = 3)
        self.down_poi_xss = second_derivative(base_array = self.down_poi_position_data,
                                          deriv_size = self.down_poi_ds, axis = 0, 
                                          ar_size = self.down_poi_N, dim = 3)
        self.up_shear_xss = second_derivative(base_array = self.up_shear_position_data,
                                          deriv_size = self.up_shear_ds, axis = 0, 
                                          ar_size = self.up_shear_N, dim = 3)
        self.down_shear_xss = second_derivative(base_array = self.down_shear_position_data,
                                          deriv_size = self.down_shear_ds, axis = 0, 
                                          ar_size = self.down_shear_N, dim = 3)
    
    def adjust_simulation_data(self):
        """
        This method adjusts all of the simulation data based on a specified time
        end point. For Poiseuille flow: t = 20; for Shear flow; t = 40. Additionally,
        this method removes the initial time steps (should be the first two), 
        when the tension profile is positive. In theory, because the filament 
        is in the compressional  quadrant, it should be negative. Therefore, 
        the instances of positive tension in the first two time steps are 
        ignored.
        """
        
        self.poi_end_time = 20
        self.shear_end_time = 40
        
        up_poi_start_idx = np.where((self.up_poi_tension_data <= 0).all(axis = 0))[0][0]
        down_poi_start_idx = np.where((self.down_poi_tension_data <= 0).all(axis = 0))[0][0]
        up_shear_start_idx = np.where((self.up_shear_tension_data <= 0).all(axis = 0))[0][0]
        down_shear_start_idx = np.where((self.down_shear_tension_data <= 0).all(axis = 0))[0][0]
        
        up_poi_end_time_idx = np.where(self.up_poi_time_data == self.poi_end_time)[0][0] + 1
        down_poi_end_time_idx = np.where(self.down_poi_time_data == self.poi_end_time)[0][0]+ 1
        up_shear_end_time_idx = np.where(self.up_shear_time_data == self.shear_end_time)[0][0]+ 1
        down_shear_end_time_idx = np.where(self.down_shear_time_data == self.shear_end_time)[0][0]+ 1
        
        # Adjust position data
        self.up_poi_position_data = self.up_poi_position_data[:,:,up_poi_start_idx:up_poi_end_time_idx]
        self.down_poi_position_data = self.down_poi_position_data[:,:,down_poi_start_idx:down_poi_end_time_idx]
        self.up_shear_position_data = self.up_shear_position_data[:,:,up_shear_start_idx:up_shear_end_time_idx]
        self.down_shear_position_data = self.down_shear_position_data[:,:,up_shear_start_idx:down_shear_end_time_idx]
        
        self.up_poi_position_adj = self.up_poi_position_adj[:,:,up_poi_start_idx:up_poi_end_time_idx]
        self.down_poi_position_adj = self.down_poi_position_adj[:,:,down_poi_start_idx:down_poi_end_time_idx]
        self.up_shear_position_adj = self.up_shear_position_adj[:,:,up_shear_start_idx:up_shear_end_time_idx]
        self.down_shear_position_adj = self.down_shear_position_adj[:,:,down_shear_start_idx:down_shear_end_time_idx]
        
        #Adjust time data
        self.up_poi_time_data = self.up_poi_time_data[up_poi_start_idx:up_poi_end_time_idx]
        self.down_poi_time_data = self.down_poi_time_data[down_poi_start_idx:down_poi_end_time_idx]
        self.up_shear_time_data = self.up_shear_time_data[up_shear_start_idx:up_shear_end_time_idx]
        self.down_shear_time_data = self.down_shear_time_data[down_shear_start_idx:down_shear_end_time_idx]
        
        # Adjust Center of Mass Data
        self.up_poi_com_vals = self.up_poi_com_vals[up_poi_start_idx:up_poi_end_time_idx,:]
        self.down_poi_com_vals = self.down_poi_com_vals[down_poi_start_idx:down_poi_end_time_idx,:]
        self.up_shear_com_vals = self.up_shear_com_vals[up_shear_start_idx:up_shear_end_time_idx,:]
        self.down_shear_com_vals = self.down_shear_com_vals[down_shear_start_idx:down_shear_end_time_idx,:]
        
        
        # Adjust True Center Data
        self.up_poi_tc_vals = self.up_poi_tc_vals[up_poi_start_idx:up_poi_end_time_idx,:]
        self.down_poi_tc_vals = self.down_poi_tc_vals[down_poi_start_idx:down_poi_end_time_idx,:]
        self.up_shear_tc_vals = self.up_shear_tc_vals[up_shear_start_idx:up_shear_end_time_idx,:]
        self.down_shear_tc_vals = self.down_shear_tc_vals[down_shear_start_idx:down_shear_end_time_idx,:]
        
        # Adjust Derivative Data
        self.up_poi_xs = self.up_poi_xs[:,:,up_poi_start_idx:up_poi_end_time_idx]
        self.down_poi_xs = self.down_poi_xs[:,:,down_poi_start_idx:down_poi_end_time_idx]
        self.up_shear_xs = self.up_shear_xs[:,:,up_shear_start_idx:up_shear_end_time_idx]
        self.down_shear_xs = self.down_shear_xs[:,:,down_shear_start_idx:down_shear_end_time_idx]
        
        self.up_poi_xss = self.up_poi_xss[:,:,up_poi_start_idx:up_poi_end_time_idx]
        self.down_poi_xss = self.down_poi_xss[:,:,down_poi_start_idx:down_poi_end_time_idx]
        self.up_shear_xss = self.up_shear_xss[:,:,up_shear_start_idx:up_shear_end_time_idx]
        self.down_shear_xss = self.down_shear_xss[:,:,down_shear_start_idx:down_shear_end_time_idx]
        
        #Normalize Time Data
        self.up_poi_time_data_adj = self.up_poi_time_data/self.up_poi_time_data.max()
        self.down_poi_time_data_adj = self.up_poi_time_data/self.up_poi_time_data.max()
        
        self.up_shear_time_data_adj = self.up_shear_time_data/self.up_shear_time_data.max()
        self.down_shear_time_data_adj = self.down_shear_time_data/self.down_shear_time_data.max()

    def create_dir(self,dir_):
        """
        This method creates the output directory if it doesn't already exist.
        """
        create_dir(dir_)
        
    
    def plot_filament_xs_transitions(self):
        """
        This method plots the phase diagram with two curves: one curve that 
        shows the instantaneous center of mass of the filament durring the
        flipping motion (U-turns) and one curve that shows the maximum value
        of the vertical component of the unit tangent vector. There are also insets
        that show a representative filament motion for each phase.
        """
        from plotting.ycom_xs_shear_up import plot_ycom_xs_shear_up
        
        file_dir = os.path.join(self.output_dir,'transition_plots/')
        self.create_dir(file_dir)
        
        # plot_center_of_mass_xs_up_poi(time_data = self.up_poi_time_data,
        #                         position_data = self.up_poi_position_adj,
        #                         center_of_mass_data = self.up_poi_com_vals[:,1],
        #                         xs_data = np.abs(self.up_poi_xs[:,1,:]).max(axis = 0),
        #                         xs_thres = 0.99,file_name = 'Poiseuille_Up',
        #                         output_dir = file_dir)
        
        # plot_center_of_mass_xs_down_poi(time_data = self.down_poi_time_data,
        #                         position_data = self.down_poi_position_adj,
        #                         center_of_mass_data = np.abs(self.down_poi_com_vals[:,1]),
        #                         xs_data = np.abs(self.down_poi_xs[:,1,:]).max(axis = 0),
        #                         xs_thres = 0.99,file_name = 'Poiseuille_Down',
        #                         output_dir = file_dir)
        
        plot_ycom_xs_shear_up(time_data = self.up_shear_time_data,
                                position_data = self.up_shear_position_adj,
                                center_of_mass_data = np.abs(self.up_shear_com_vals[:,1]),
                                xs_data = np.abs(self.up_shear_xs[:,1,:]).max(axis = 0),
                                xs_thres = 0.99,file_name = 'Shear_Up',
                                output_dir = file_dir)
        
        # plot_center_of_mass_xs_down_shear(time_data = self.down_shear_time_data,
        #                         position_data = self.down_shear_position_adj,
        #                         center_of_mass_data = np.abs(self.down_shear_com_vals[:,1]),
        #                         xs_data = np.abs(self.down_shear_xs[:,1,:]).max(axis = 0),
        #                         xs_thres = 0.99,file_name = 'Shear_Down',
        #                         output_dir = file_dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_directory",
                        help="Specify the directory where the main script and accompany scripts are located in",
                    type = str)
    parser.add_argument("input_poiseuille_directory",
                        help="Specify the main directory of the Poiseuille flow Non-Brownian Simulations",
                    type = str)
    parser.add_argument("input_shear_directory",
                        help="Specify the main directory of the Poiseuille flow Non-Brownian Simulations",
                    type = str)
    parser.add_argument("output_directory",
                        help="Specify the parent directory of the Migration Data",
                    type = str)
    # args = parser.parse_args()
    args = parser.parse_args(['C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01b_Non_Brownian_Analysis\\',
                              'C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01_Migration_Simulations\\02_Actual_Results\\NB_Poiseuille_J\\UC_1p00_MB_1p00e5\\',
                              'C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01_Migration_Simulations\\02_Actual_Results\\NB_Shear_J\\MB_1p00e5\\',
                              'C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01b_Non_Brownian_Analysis\\01_Actual_Results\\Shear_Poiseuille_Transitions\\'])
        
    os.chdir(args.project_directory)
    #Assume calculations/ and misc/ is in the project_directory
    from misc.create_dir import create_dir
    from calculations.center_of_mass import center_of_mass
    from calculations.adjust_position_data import adjust_position_data
    from calculations.first_derivative import first_derivative
    from calculations.second_derivative import second_derivative
    from calculations.true_center import true_center
    
    
    ### Up and Down Data should be in separate directories under parent directory
    poi_shear_comp = poi_shear_data(up_poi_dir = os.path.join(args.input_poiseuille_directory,'Up_0p30/'),
                                    down_poi_dir = os.path.join(args.input_poiseuille_directory,'Down_0p30/'),
                                    up_shear_dir = os.path.join(args.input_shear_directory,'Up_0p30/'),
                                    down_shear_dir = os.path.join(args.input_shear_directory,'Down_0p30/'),
                                    output_dir = args.output_directory)
    
    poi_shear_comp.read_poi_data()
    poi_shear_comp.read_shear_data()
    poi_shear_comp.adjust_positions()
    poi_shear_comp.calc_center_of_mass()
    poi_shear_comp.calc_true_center()
    poi_shear_comp.calc_derivatives()
    poi_shear_comp.adjust_simulation_data()
    poi_shear_comp.plot_filament_xs_transitions()


#%%
# tension = poi_shear_comp.up_shear_tension_data.copy()
# xs = np.abs(poi_shear_comp.up_poi_xs.copy()[:,1,:]).max(axis = 0)
# com = np.abs(poi_shear_comp.up_poi_com_vals.copy()[:,1])
# time_data = poi_shear_comp.up_poi_time_data.copy()


# ## Remove first couple of instances
# start_idx = 2
# xs = xs[start_idx:]
# com = com[start_idx:]
# time_data = time_data[start_idx:]

# fig,axes = plt.subplots(figsize = (7,7))
# ax1 = axes.twinx()
# ax1.plot(time_data,xs,color = 'red')
# axes.plot(time_data,com,color = 'blue')
# plt.show()