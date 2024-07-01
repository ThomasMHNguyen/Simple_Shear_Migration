# -*- coding: utf-8 -*-
"""
FILE NAME:      A__v01_02_Process_COM_Migration_Data.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        create_dir.py; Plot_Ensemble_Time_Trajectory; Plot_Ensemble_Avg_Time_Trajectory;
                Plot_Ensemble_COM_Distribution.py; Plot_Ensemble_Avg_Velocity_Fits.py;
                Plot_Depletion_Layer_Thickness.py; Plot_COM_Movement_Fitted_Parameters.py;
                

DESCRIPTION:    This script will read in all of the simulation data files regarding
                shear-induced migration in either shear or Poiseuille flow and process
                it to calculate ensemble averages of the center of mass. It will plot the average trajectory
                of the filament in fluid flow, average stress tensor values, drift velocity, and 
                plots the relationship between net displacement and flow strength.

INPUT
FILES(S):       1) .NPY file that contains all positions of each discretized
                position of the filament for the duration of the simulation.
                2) .NPY file that contains the stresses of the filament at each 
                timestep. 
                3) .CSV file that lists all parameters used for run. 

OUTPUT
FILES(S):       1) .PNG file that shows the trajectory of each ensemble simulation
                grouped by parameter values. 
                
                2) .PNG file that shows the average ensemble simulation trajectory 
                grouped by parameter values. 
                
                3) .PNG file that shows the relationship between depletion layer
                thickness and flow strength. 


INPUT
ARGUMENT(S):    1) Main Input directory: The directory that will houses all of the 
                simulation data.
                1) Main Output directory: The directory that will house all of the
                output files associated with the analysis; if it doesn't exist,
                it will be created.

CREATED:        14Jul22

MODIFICATIONS
LOG:
14Sep22         1) Added functionality to calculate ensemble averages of stress values,
                and drift velocity based on average displacement.
14Sep22         2) Added functionality to plot average stress behavior, and drift 
                velocity.
14Sep22         3) Removed intermediary DataFrame that stores data pertaining
                to a particular ensemble.
23Sep22         4) Added code to determine the elastic energy peak. Added code 
                to calculate filament curvature. 
22Nov22         5) Renamed from B__v01_02_Process_Migration_data.py to 
                A__v01_03_Process_COM_Migration_data.py. Simplified code by 
                removing features to identify instances of bending and 
                calculating curvature. These instances will be part of another script. 
22Nov22         6) Migrated Plotting functions to a separate code. 
23Apr23         7) Reorganized dependency code in the project directory. 
23Apr23         8) Removed extra particle stress and elastic energy data from 
                being saved.  
26Sep23         9) Restructured class methods.

    
            
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.9.16

VERSION:        1.4

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:     

NOTE(S):        N/A

"""
import re, os, argparse, logging, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})


class indiv_ensemble_data():
    """
    This class reads in the data files pertaining to a particular ensemble.
    """
    
    def __init__(self,path_to_files,rep_count):
        """
        The initialization function reads in all of the data files and pre-calculates
        the center of mass for the filament.
        
        Inputs:
            
        path_to_files:          Absolute path to data files of an ensemble.
        rep_count:              Ensemble number.
        """
        #Read in the associated 
        self.dir_ = path_to_files
        self.position_data = np.load(os.path.join(self.dir_,'filament_allstate.npy'))
        self.position_data[:,0,:] = self.position_data[:,0,:] - self.position_data[:,0,:].mean(axis = 0)#Adjust for any translation in x-coordinates
        self.params_df = pd.read_csv(os.path.join(self.dir_,'parameter_values.csv'),index_col = 0,
                                     header = 0)
        self.time_data = np.load(os.path.join(self.dir_,'filament_time_vals_all.npy'))
        self.elastic_data = np.load(os.path.join(self.dir_,'filament_elastic_energy.npy'))
        self.center_mass = self.position_data.mean(axis = 0).T #Rows are each time point, columns are x,y,z, components
        self.rep_number = rep_count
        
    def find_ensemble_parameters(self):
        """
        This function reads the parameter CSV file and saves the important parameters.
        """
        self.N = int(self.params_df.loc['N','Value'])
        self.ds = 1/(self.N-1)
        self.dt = float(self.params_df.loc['Array Time Step Size','Value'])
        self.rigidity_profile = self.params_df.loc['Rigidity Function Type','Value']
        self.mu_bar = int(self.params_df.loc['Mu_bar','Value'])
        self.channel_height = float(self.params_df.loc['Channel Upper Height','Value'])
        self.U_centerline = float(self.params_df.loc['Poiseuille U Centerline','Value'])
        self.vert_displ = float(self.params_df.loc['Vertical Displacement','Value'])
        self.velo_exp = int(self.params_df.loc['Steric Velocity Exponential Coefficient','Value'])
        self.velo_gap = float(self.params_df.loc['Steric Velocity Gap Criteria','Value'])
        self.s_start = float(self.params_df.loc['Filament s start','Value'])
        self.s_end = float(self.params_df.loc['Filament s end','Value'])
        self.s = np.linspace(self.s_start,self.s_end,self.N)
        self.true_center_loc = np.where(self.s == 0)[0][0]
        self.flow_type = self.params_df.loc['Flow Type','Value']
        self.sterics_use = self.params_df.loc['Sterics Use','Value']
        self.brownian_use = self.params_df.loc['Brownian Use','Value']
        if self.params_df.loc['Kolmogorov Phase','Value'] == 'Pi':
            self.kflow_phase_text = 'Pi'
            self.kflow_phase_val = np.pi
        else:
            self.kflow_phase_text = '{:.2f}'.format(float(self.params_df.loc['Kolmogorov Phase','Value']))
            self.kflow_phase_val = float(self.params_df.loc['Kolmogorov Phase','Value'])
        self.kflow_freq = float(self.params_df.loc['Kolmogorov Frequency','Value'])
        self.true_center = self.position_data[self.true_center_loc,:,:] #rows are x,y,z components, columns are each time point
        
        logging.info(
            "Now finished reading in ensemble data for Mu_bar = {} | Replicate = {} | y_0 = {} | Flow Type: {} | {} Sterics | {} Brownian use | H = {} | Poi Uc = {} | K-Flow n = {} | K-Flow phi = {}".format(
                self.mu_bar,self.rep_number,
                self.vert_displ,self.flow_type,
                self.sterics_use,self.brownian_use,
                self.channel_height,self.U_centerline,
                self.kflow_freq,self.kflow_phase_text))   


class all_ensemble_data():
    """
    This class will store all of the ensemble data temporarily in a dictionary
    before converting it into a DataFrame.
    """
    
    
    def __init__(self,output_dir):
        """
        This initialization method will just create an empty list for all ensemble
        data to be stored in.
        """
        self.all_ensemble_loc_data_list = []
        self.drift_velocity_exp_eqn = lambda t,a,b,t_0: a + b*np.exp(t/t_0)
        self.output_dir = output_dir
    
    def add_position_stress_data(self,ensemble_class):
        """
        This method will convert each individual ensemble data (present in a DataFrame)
        into a dictionary before appending it to the list. This function specifically pertains
        to positional data and stress data.
        
        Inputs:
        
        ensemble_class:         Class variable that has all of the information pertaining
                                to a particular ensemble.
        """
        self.ensemble_data_dict = {'Rigidity Suffix':ensemble_class.rigidity_profile,
                                   'Mu_bar': ensemble_class.mu_bar,
                                   's Start': ensemble_class.s_start,
                                   's End': ensemble_class.s_end,
                                   'N': ensemble_class.N,
                              'Brownian Time': ensemble_class.time_data,
                              'Adjusted Time':ensemble_class.mu_bar*ensemble_class.time_data,
                              'Rep Number': ensemble_class.rep_number,
                              'Channel Height': ensemble_class.channel_height,
                              'Poiseuille U Centerline': ensemble_class.U_centerline,
                              'Kolmogorov Phase Text': ensemble_class.kflow_phase_text,
                              'Kolmogorov Phase Value': ensemble_class.kflow_phase_val,
                              'Kolmogorov Frequency': ensemble_class.kflow_freq,
                              'Starting Vertical Displacement': ensemble_class.vert_displ,
                              'Steric Velocity Exponential Coefficient':ensemble_class.velo_exp,
                              'Steric Velocity Gap Criteria': ensemble_class.velo_gap, 
                              "Flow Type": ensemble_class.flow_type,
                              "Sterics Use": ensemble_class.sterics_use,
                              "Brownian Use": ensemble_class.brownian_use,
                              'Center of Mass-x': ensemble_class.center_mass[:,0],
                              'Center of Mass-y': ensemble_class.center_mass[:,1],
                              'Center of Mass-z': ensemble_class.center_mass[:,2],
                              'True Center-x': ensemble_class.true_center[0,:],
                              'True Center-y': ensemble_class.true_center[1,:],
                              'True Center-z': ensemble_class.true_center[2,:],
                              "SQ Net Center of Mass-y":(ensemble_class.center_mass[:,1]-ensemble_class.vert_displ)**2}
        self.all_ensemble_loc_data_list.append(self.ensemble_data_dict)
        
                   
    def compile_position_stress_data(self):
        """
        This method converts the dictionary of all ensemble positional and 
        stress data into a Pandas DataFrame.
        """
        self.all_ensemble_loc_data_df = pd.concat(
            [pd.DataFrame.from_dict(i) for i in self.all_ensemble_loc_data_list],ignore_index = True)
        
        
    def calculate_average_com_position(self):
        """
        This method calculates the ensemble averages of the center of mass 
        displacement based on a given flow strength, channel height, 
        starting displacement, and centerline velocity.
        """
        
        cols_ignore = ['Center of Mass-x','Center of Mass-y','Center of Mass-z',
                       "SQ Net Center of Mass-y",'True Center-x','True Center-y',
                       'True Center-z','Rep Number','Adjusted Time']
        com_data_cols = cols_ignore[:4]

        
        self.avg_com_y_displace_df = self.all_ensemble_loc_data_df.groupby(
            by = [i for i in self.all_ensemble_loc_data_df.columns if i not in cols_ignore])[com_data_cols].mean()
        self.avg_com_y_displace_df = self.avg_com_y_displace_df.reset_index()
        
        self.avg_com_y_displace_df['ABS Center of Mass-y'] = np.abs(self.avg_com_y_displace_df['Center of Mass-y'])
        self.all_ensemble_loc_data_df['ABS Center of Mass-y'] = np.abs(self.all_ensemble_loc_data_df['Center of Mass-y'])
    
    def calculate_average_tc_position(self):
        """
        This method calculates the ensemble averages of the true center 
        displacement, based on a given flow strength, channel height, 
        starting displacement, and centerline velocity.
        """
        cols_ignore = ['Center of Mass-x','Center of Mass-y','Center of Mass-z',
                       "SQ Net Center of Mass-y",'True Center-x','True Center-y',
                       'True Center-z','Rep Number','Adjusted Time']
        tc_data_cols = cols_ignore[4:7]
        self.avg_true_y_displace_df = self.all_ensemble_loc_data_df.groupby(
            by = [i for i in self.all_ensemble_loc_data_df.columns if i not in cols_ignore])[tc_data_cols].mean()
        self.avg_true_y_displace_df = self.avg_true_y_displace_df.reset_index()
    
    
    def calculate_ensemble_depletion_layer(self):
        """
        This method calculates the distance between the final y-position of the 
        filament and the wall for all ensembles.
        """
        exp_groups = self.all_ensemble_loc_data_df.groupby(
            by = ['Rigidity Suffix','Mu_bar','Channel Height','Rep Number','Starting Vertical Displacement',  #0-4
                  'Poiseuille U Centerline','Kolmogorov Phase Text','Kolmogorov Phase Value','Kolmogorov Frequency', #5-8
                  'Steric Velocity Exponential Coefficient','Steric Velocity Gap Criteria', #9-10
                  'Sterics Use','Flow Type']) #11-12
        for group in exp_groups.groups.keys():
            group_df = exp_groups.get_group(group)
            
            final_time_df = group_df[group_df['Brownian Time'] == group_df['Brownian Time'].max()]
            dist_from_wall = (final_time_df['Channel Height'] - final_time_df['ABS Center of Mass-y']).to_numpy()[0]
            self.all_ensemble_loc_data_df.loc[group_df.index.values,'Distance From Wall'] = np.abs(dist_from_wall)
            
            
    def calculate_avg_com_depletion_layer(self):
        """
        This method calculates the distance between the final y-position of the 
        filament and the wall based on the average ensemble data.
        """
        exp_groups = self.avg_com_y_displace_df.groupby(
            by = ['Rigidity Suffix','Mu_bar','Channel Height','Starting Vertical Displacement',  #0-3
                  'Poiseuille U Centerline','Kolmogorov Phase Text','Kolmogorov Phase Value','Kolmogorov Frequency', #4-7
                  'Steric Velocity Exponential Coefficient','Steric Velocity Gap Criteria', #8-9
                  'Sterics Use','Flow Type']) #10-11
        for group in exp_groups.groups.keys():
            group_df = exp_groups.get_group(group)
            final_time_df = group_df[group_df['Brownian Time'] == group_df['Brownian Time'].max()]
            dist_from_wall = (final_time_df['Channel Height'] - final_time_df['ABS Center of Mass-y']).to_numpy()[0]
            self.avg_com_y_displace_df.loc[group_df.index.values,'Distance From Wall'] = np.abs(dist_from_wall    )
            
    def filter_data(self,flow_type):
        """
        This method filters for removes all data where the ensembles initially start 
        at y = 0.

        """
        if flow_type == 'Poiseuille' or flow_type == 'Shear':
            self.avg_com_y_displace_df = self.avg_com_y_displace_df[
                self.avg_com_y_displace_df['Starting Vertical Displacement'] > 0]
            
            self.all_ensemble_loc_data_df = self.all_ensemble_loc_data_df[
                self.all_ensemble_loc_data_df['Starting Vertical Displacement'] > 0]
        
        elif flow_type == 'Kolmogorov':
            vals_choose = np.round(np.arange(0,2 + 0.1,0.1),2)
            self.avg_com_y_displace_df = self.avg_com_y_displace_df[
                self.avg_com_y_displace_df['Starting Vertical Displacement'].isin(vals_choose)]
            
            self.all_ensemble_loc_data_df = self.all_ensemble_loc_data_df[
                self.all_ensemble_loc_data_df['Starting Vertical Displacement'].isin(vals_choose)]
            

    def fit_drift_velocity_exp_eqn(self):
        """
        This method performs a curve-fitting based on the average center of mass-y
        data to an exponential plateau equation of the form y = a+b*e(t/t_0).
        The parameters are then saved to a new DataFrame. 
        """
        
        exp_groups = self.avg_com_y_displace_df.groupby(
            by = ['Rigidity Suffix','Mu_bar','Channel Height','Starting Vertical Displacement',  #0-3
                  'Poiseuille U Centerline','Kolmogorov Phase Text','Kolmogorov Phase Value','Kolmogorov Frequency', #4-7
                  'Steric Velocity Exponential Coefficient','Steric Velocity Gap Criteria', #8-9
                  'Sterics Use','Flow Type']) #10-11
        
        for group in exp_groups.groups.keys():
            group_df = exp_groups.get_group(group)
            rigid,mu_bar,channel_h,vert_displ,u_centerline,k_phase_text,k_phase_val,k_freq,velo_exp,gap_criteria = group[:10]
            steric_use,flow_type = group[-2:]
            
            """Method 1"""
            # guess_vals_1 = np.array([(channel_h/2),vert_displ - (channel_h/2),-0.1]) #for starting vertical displacements near the wall
            
            # fit_vals_1,fit_cov_matr_1 = curve_fit(f = self.drift_velocity_exp_eqn,xdata = np.array(group_df['Brownian Time']),
            #                        ydata = np.array(group_df['ABS Center of Mass-y']),
            #                        p0 = guess_vals_1)
            
            # choose_fit_vals = fit_vals_1
          
            """Method 2"""
            guess_vals_1 = np.array([(channel_h/2) - vert_displ,(channel_h/2),-0.1]) #for starting vertical displacements near the wall
            guess_vals_2 = np.array([(channel_h/2) +vert_displ,- (channel_h/2),-0.1])
            
            fit_vals_1,fit_cov_matr_1 = curve_fit(f = self.drift_velocity_exp_eqn,xdata = np.array(group_df['Brownian Time']),
                                   ydata = np.array(group_df['ABS Center of Mass-y']),
                                   p0 = guess_vals_1)
            fit_vals_2,fit_cov_matr_2 = curve_fit(f = self.drift_velocity_exp_eqn,xdata = np.array(group_df['Brownian Time']),
                                    ydata = np.array(group_df['ABS Center of Mass-y']),
                                    p0 = guess_vals_2)
            
            ## Choose fit parameters with smallest covariant matrix ###
            if np.linalg.norm(np.sqrt(np.diag(fit_cov_matr_1))) < np.linalg.norm(np.sqrt(np.diag(fit_cov_matr_2))):
                choose_fit_vals,choose_fit_std = fit_vals_1,np.sqrt(np.diag(fit_cov_matr_1))
            elif np.linalg.norm(np.sqrt(np.diag(fit_cov_matr_1))) > np.linalg.norm(np.sqrt(np.diag(fit_cov_matr_2))):
                choose_fit_vals,choose_fit_std = fit_vals_2,np.sqrt(np.diag(fit_cov_matr_2))
            else:
                choose_fit_vals,choose_fit_std = fit_vals_1,np.sqrt(np.diag(fit_cov_matr_1))
            
            self.avg_com_y_displace_df.loc[group_df.index.values,'A'] = choose_fit_vals[0]
            self.avg_com_y_displace_df.loc[group_df.index.values,'B'] = choose_fit_vals[1]
            self.avg_com_y_displace_df.loc[group_df.index.values,'T_0'] = choose_fit_vals[2]
            self.avg_com_y_displace_df.loc[group_df.index.values,'A STDEV'] = choose_fit_std[0]
            self.avg_com_y_displace_df.loc[group_df.index.values,'B STDEV'] = choose_fit_std[1]
            self.avg_com_y_displace_df.loc[group_df.index.values,'T_0 STDEV'] = choose_fit_std[2]
            self.avg_com_y_displace_df.loc[group_df.index.values,'Fitted Drift Data'] = choose_fit_vals[0] + choose_fit_vals[1]*np.exp(group_df['Brownian Time']/choose_fit_vals[2])
    
    def height_class(self,vert_displ):
        """
        This method gives a string classifier to the average center of mass
        data based on its initial condition.
        
        Inputs:
            
        vert_displ:         Initial condition of the filament.
        """
        if vert_displ < 0.20:
            return 'Low'
        elif vert_displ >= 0.20 and vert_displ <= 0.30:
            return 'Middle'
        elif vert_displ > 0.30 and vert_displ <= 0.45:
            return 'High'
        
    def apply_height_class(self):
        """
        This method gives a string classifier to the average center of mass
        data based on its initial condition and removes any N/A data.
    
        """
        self.avg_com_y_displace_df['Height Class'] = self.avg_com_y_displace_df['Starting Vertical Displacement'].apply(lambda x: self.height_class(x))
        self.fil_avg_com_y_displace_df = self.avg_com_y_displace_df.dropna(how = 'any')
        self.filter_for_fit_params()
        
    def filter_for_fit_params(self):
        """
        This method keeps the classifier columns and Parameters & standard 
        deviations from the curve fitting.
        """
        self.filtered_fit_params_df = self.avg_com_y_displace_df.copy().filter(items = [
            'Rigidity Suffix','Mu_bar','Channel Height','Starting Vertical Displacement',
            'Poiseuille U Centerline','Kolmogorov Phase Text','Kolmogorov Phase Value','Kolmogorov Frequency', 
            'Steric Velocity Exponential Coefficient','Steric Velocity Gap Criteria', 
            'Sterics Use','Flow Type','Brownian Time','Height Class','A','B','T_0'])
        self.filtered_fit_params_df = pd.melt(self.filtered_fit_params_df,
                                          id_vars = self.filtered_fit_params_df.columns.values[:-3],
                                          value_vars = self.filtered_fit_params_df.columns.values[-3:],
                                          var_name = 'Fit Parameter',
                                          value_name = 'Value')
        
        self.filtered_fit_stdev_params_df = self.avg_com_y_displace_df.copy().filter(items = [
            'Rigidity Suffix','Mu_bar','Channel Height','Starting Vertical Displacement',
            'Poiseuille U Centerline','Kolmogorov Phase Text','Kolmogorov Phase Value','Kolmogorov Frequency', 
            'Steric Velocity Exponential Coefficient','Steric Velocity Gap Criteria', 
            'Sterics Use','Flow Type','Brownian Time','Height Class','A STDEV','B STDEV','T_0 STDEV'])
        self.filtered_fit_stdev_params_df = pd.melt(self.filtered_fit_stdev_params_df,
                                          id_vars = self.filtered_fit_stdev_params_df.columns.values[:-3],
                                          value_vars = self.filtered_fit_stdev_params_df.columns.values[-3:],
                                          var_name = 'Fit Parameter',
                                          value_name = 'Value')
    
    def fit_trajectory_depl_layer(self):
        """
        This method creates the scaling relationship between depletion layer and mu_bar
        by extracting the final position of the average center of mass and fitting
        it to the relationship L_d ~ mu_bar^(-1/8).
        
        """
        ### Fit to the curve H - <y_com_f> ~ mu_bar^(-1/8) or H - y_com_f ~ mu_bar^(-1/8)
        def ld_scaling(x,a): return a*x**(-0.125)
        # def ld_scaling(x,a,b): return a*x**(b)
        
        
        
        self.avg_com_copy_df = self.avg_com_y_displace_df.copy()
        
        #Filter out average trajectories based on whether or not they migrated
        self.idx_to_ignore = []
        self.weak_2e4_remove_idx = self.avg_com_copy_df[(self.avg_com_copy_df['Mu_bar'] == 2.5e4) &\
                                    ((self.avg_com_copy_df['Starting Vertical Displacement'] == 0.05) |\
                                     (self.avg_com_copy_df['Starting Vertical Displacement'] == 0.10))].index.values
        self.strong_2e5_remove_idx = self.avg_com_copy_df[(self.avg_com_copy_df['Mu_bar'] == 2e5) &\
                                    (self.avg_com_copy_df['Starting Vertical Displacement'] == 0.20)].index.values
        self.idx_to_ignore.extend(self.weak_2e4_remove_idx)
        self.idx_to_ignore.extend(self.strong_2e5_remove_idx)
        self.avg_com_copy_df = self.avg_com_copy_df[~self.avg_com_copy_df.index.isin(self.idx_to_ignore)]
        self.avg_com_copy_df = self.avg_com_copy_df[self.avg_com_copy_df['Brownian Time'] == 5e-2]
        
            
        # #Method 1: Calculate regression & extract slope
        # ## Log10 transformation ##
        # self.avg_com_copy_df['Log10 Mu_bar'] = np.log10(self.avg_com_copy_df['Mu_bar'])
        # self.avg_com_copy_df['Log10 Distance From Wall'] = np.log10(self.avg_com_copy_df['Distance From Wall'])
        
        # ### Calculate Linear Regression between Displacement and mu_bar ###
        # linear_model = smf.ols('Q("Log10 Distance From Wall") ~ Q("Log10 Mu_bar")',data = self.avg_com_copy_df).fit()
        # intercept,slope,rsquared = 10**linear_model.params[0],linear_model.params[1],linear_model.rsquared
        
        #Method 2: Fit to curve of H - <y_com_f> ~ mu_bar^(-1/8)
        fit_params,_ = curve_fit(ld_scaling,xdata = self.avg_com_copy_df['Mu_bar'].to_numpy(),ydata = self.avg_com_copy_df['Distance From Wall'].to_numpy())
        #Parameters to plot Trajectory data and depletion layer scaling
        self.param_dict = {"Mu_bar to Plot 1": 0,
                      "Mu_bar to Plot 2": 0,
                      # "Fit Slope": fit_params[1],
                      "Fit Slope": -0.125,
                      "Fit Intercept": fit_params[0],
                      "Channel Height":self.avg_com_y_displace_df['Channel Height'].unique()[0],
                      "Poiseuille U Centerline":self.avg_com_y_displace_df['Poiseuille U Centerline'].unique()[0]}

    def create_dir(self,dir_):
        """
        This method imports the create_dir.py script in order to create 
        directories if they dont' exist.
        """
        from misc.create_dir import create_dir
        create_dir(dir_)
        
    def center_of_mass_ensemble_plot(self):
        """
        This method imports the ensemble_ycom_time_data.py script in order
        to plot the center of mass trajectories of all ensembles.
        """
        from plotting.ensemble_ycom_time_data import ensemble_ycom_time_data
        
        new_output_dir = os.path.join(self.output_dir,'com_ensbl_data/')
        self.create_dir(new_output_dir)
        
        ensemble_ycom_time_data(
            ensemble_data_df = self.all_ensemble_loc_data_df,
            output_dir = new_output_dir)
        
    def true_center_ensemble_plot(self):
        """
        This method imports the ensemble_ytc_time_data.py script in order
        to plot the true center trajectories of all ensembles.
        """
        from plotting.ensemble_ytc_time_data import ensemble_ytc_time_data      
        new_output_dir = os.path.join(self.output_dir,'tc_ensbl_data/')
        self.create_dir(new_output_dir)
        ensemble_ytc_time_data(
            ensemble_data_df = self.all_ensemble_loc_data_df,
            output_dir = new_output_dir)
        
        
    def avg_center_of_mass_ensemble_plot(self):
        """
        This method imports the ensemble_avg_ycom_time_data.py script in order
        to plot the average ensemble center of mass trajectory.
        """
        from plotting.ensemble_avg_ycom_time_data import ensemble_avg_ycom_time_data
        new_output_dir = os.path.join(self.output_dir,'avg_com/')
        self.create_dir(new_output_dir)
        ensemble_avg_ycom_time_data(
            ensemble_data_df = self.avg_com_y_displace_df,
            output_dir = new_output_dir)
        
       
    def avg_true_center_ensemble_plot(self):
        """
        This method imports the Plot_Ensemble_Avg_Time_Trajectory.py script in order
        to plot the average ensemble true center trajectory.
        """
        from plotting.ensemble_avg_ytc_time_data import ensemble_avg_ytc_time_data
        new_output_dir = os.path.join(self.output_dir,'avg_tc/')
        self.create_dir(new_output_dir)
        ensemble_avg_ytc_time_data(
            ensemble_data_df = self.avg_true_y_displace_df,
            output_dir = new_output_dir)
    
        
    def ensemble_avg_ycom_traj_fits(self):
        """
        This method imports the Plot_Ensemble_Avg_Velocity_Fits.py script in order
        to plot both the average ensemble center of mass trajectory data and
        the fitted data based on the exponential plateau equation. 
        """
        from plotting.ensemble_avg_ycom_traj_fits import ensemble_avg_ycom_traj_fits
        
        ## Plot Drift Velocity Fits vs. Actual Data ###
        new_output_dir = os.path.join(self.output_dir,'COM_Traj/')
        self.create_dir(new_output_dir)
        ensemble_avg_ycom_traj_fits(ensemble_data_df = self.avg_com_y_displace_df,
                                output_dir = new_output_dir)
        
        
    def plot_avg_com_fits_params(self):
        """
        This method imports the Plot_COM_Movement_Fitted_Parameters.py script
        in order to plot the fitted parameter values as a function of mu_bar.
        """
        from plotting.plot_avg_com_fits_params import plot_avg_com_fits_params
        
        new_output_dir = os.path.join(self.output_dir,'com_Exp_Plateau_Eq_Params/')
        self.create_dir(new_output_dir)
        plot_avg_com_fits_params(ensemble_data_df = self.filtered_fit_params_df,
                                output_dir = new_output_dir)
        
        
    def depl_layer_thickness_plot(self,data_type):
        """
        This method imports the depletion_layer.py script in order
        to plot the distance between the filament filament position and the wall
        (depletion layer thickness).
        
        Inputs:
            
        data_type:      String that specifies whether to plot the depletion layer
                        scaling based on all ensembles or average ensemble data.
        """
        from plotting.depletion_layer import depletion_layer
        new_output_dir = os.path.join(self.output_dir,'mu_bar_depl_layer/')
        self.create_dir(new_output_dir)
        
        ### Fit to the curve H - <y_com_f> ~ mu_bar^(-1/8) or H - y_com_f ~ mu_bar^(-1/8)
        def ld_scaling(x,a,b): return a*x**(b)
        # def ld_scaling(x,a,b): return a*x**(b)
        
        
        if data_type == 'average_com':
            fil_df = self.avg_com_y_displace_df.copy()

            """Method 1: Calculate regression"""
            # ## Log10 transformation ##
            # fil_df['Log10 Mu_bar'] = np.log10(fil_df['Mu_bar'])
            # fil_df['Log10 Distance From Wall'] = np.log10(fil_df['Distance From Wall'])
            
            # ### Calculate Linear Regression between Displacement and mu_bar ###
            # linear_model = smf.ols('Q("Log10 Distance From Wall") ~ Q("Log10 Mu_bar")',data = fil_df).fit()
            # intercept,slope,rsquared = linear_model.params[0],linear_model.params[1],linear_model.rsquared
            
            """Method 2: Fit to curve of H - <y_com_f> ~ mu_bar^(-1/8)"""
            fit_params,_ = curve_fit(ld_scaling,xdata = fil_df['Mu_bar'].to_numpy(),ydata = fil_df['Distance From Wall'].to_numpy())
            
            fil_df = fil_df[(fil_df['Brownian Time'] == 5e-2) & (fil_df['Channel Height'] == 0.25)]
            
            params_dict = {"Slope":fit_params[1],
                           "Intercept": fit_params[0]}
            
            
            depletion_layer(
                ensemble_avg_data_df = fil_df,
                params_dict = params_dict,
                data_type = data_type,
                file_name = 'Depl_thick_avg',
                output_directory = new_output_dir)
            
            print("All Ensemble: Slope = {0:.3f} | Intercept = {1:.3f}".format(fit_params[1],fit_params[0]))
        elif data_type == 'ensemble_com':
            fil_df = self.all_ensemble_loc_data_df.copy()
            
            """ Method 1: Calculate regression"""
            # ## Log10 transformation ##
            # fil_df['Log10 Mu_bar'] = np.log10(fil_df['Mu_bar'])
            # fil_df['Log10 Distance From Wall'] = np.log10(fil_df['Distance From Wall'])
            
            # ### Calculate Linear Regression between Displacement and mu_bar ###
            # linear_model = smf.ols('Q("Log10 Distance From Wall") ~ Q("Log10 Mu_bar")',data = fil_df).fit()
            # intercept,slope,rsquared = linear_model.params[0],linear_model.params[1],linear_model.rsquared
            
            """Method 2: Fit to curve of H - <y_com_f> ~ mu_bar^(-1/8)"""
            fit_params,_ = curve_fit(ld_scaling,xdata = fil_df['Mu_bar'].to_numpy(),ydata = fil_df['Distance From Wall'].to_numpy())
            
            fil_df = fil_df[(fil_df['Brownian Time'] == 5e-2) & (fil_df['Channel Height'] == 0.25)]
            
            params_dict = {"Slope":fit_params[1],
                           "Intercept": fit_params[0]}
            
            depletion_layer(
                ensemble_avg_data_df = fil_df,
                params_dict = params_dict,
                data_type = data_type,
                file_name = 'Depl_thick',
                output_directory = new_output_dir)
            
            print("Average Ensemble: Slope = {0:.3f} | Intercept = {1:.3f}".format(fit_params[1],fit_params[0]))
            
        
        
    def ensemble_com_distr_k_flows(self):
        """
        This method plots the y-center of mass distribution values of all ensembles at
        specific timepoints in Kolmogorov flow.
        
        Inputs:
            
        flow_type:      String that specifies what type of flow to plot the 
                        distribution data for.
        """
        from plotting.ycom_distr_snps_k_flows import ycom_distr_snps_k_flows
        new_output_dir = os.path.join(self.output_dir,'com_distr/')
        self.create_dir(new_output_dir)
        
        fil_df = self.all_ensemble_loc_data_df.copy()
        ycom_distr_snps_k_flows(
            ensemble_data_df = fil_df,
            time_vals = np.array([0,5e-3,1e-1,1.5e-1,2e-1]),
            time_vals_text = [r"0",r"5.0 \times 10^{-3}",r"1.0 \times 10^{-1}",
                                r"1.5 \times 10^{-1}",r"2.0 \times 10^{-1}"],
            output_dir = new_output_dir)
        
            
    def ensemble_com_distr_poiseuille(self,mu_bar1,mu_bar2):
        """
        This method plots the y-center of mass distribution values of all ensembles at
        specific timepoints in Poiseuille flow.
        
        Inputs:
            
        mu_bar_1:                       One of the Mu_bar values to plot the 
                                        data for.
        mu_bar_2:                       One of the Mu_bar values to plot the 
                                        data for.
        """
        from plotting.ycom_distr_snps_poi import ycom_distr_snps_poi
        
        new_output_dir = os.path.join(self.output_dir,'com_distr_combined/')
        self.create_dir(new_output_dir)

        ## Poiseuille/Shear
        ycom_distr_snps_poi(
            ensemble_data_df = self.all_ensemble_loc_data_df,
            mu_bar_1 = mu_bar1,mu_bar_2 = mu_bar2,
            time_vals = np.array([0,1.5e-2,3e-2,4.5e-2,5e-2]),
            time_vals_text = [r"0",r"1.5 \times 10^{-2}",r"3.0 \times 10^{-2}",
                              r"4.5 \times 10^{-2}",r"5.0 \times 10^{-2}"],
            output_dir = new_output_dir)
            
        
    def com_trajectory_depletion_layer(self,data_type,mu_bar1,mu_bar2):
        from plotting.Plot_Depletion_Layer_Thickness_COM_Traj_subplots import plot_com_traj_depl_layer_thickness
        
        new_output_dir = os.path.join(self.output_dir,'COM_Traj_Depl/')
        self.create_dir(new_output_dir)
        
        
        self.param_dict["Mu_bar to Plot 1"] = mu_bar1
        self.param_dict["Mu_bar to Plot 2"] = mu_bar2
            
        
        plot_com_traj_depl_layer_thickness(ensemble_avg_data_df = self.avg_com_y_displace_df,
                                           ld_fit_df = self.avg_com_copy_df,
                                           parameter_dict = self.param_dict,
                                           data_type = data_type,
                                           file_name = 'COM_Traj_Depl_subplots',
                                           output_directory = new_output_dir)
        
    def ycom_traj_vault_snp_depl_layer(self,data_type,mu_bar1,mu_bar2,
                                                    snp_ar1,snp_ar2,time_data_1,
                                                    time_data_2):
        from plotting.ycom_traj_vault_snp_depl_layer import ycom_traj_vault_snp_depl_layer
        
        new_output_dir = os.path.join(self.output_dir,'COM_Traj_Vault_Depl/')
        self.create_dir(new_output_dir)
        
        self.param_dict["Mu_bar to Plot 1"] = mu_bar1
        self.param_dict["Mu_bar to Plot 2"] = mu_bar2
        
        ycom_traj_vault_snp_depl_layer(ensemble_avg_data_df = self.avg_com_y_displace_df,
                                      ld_fit_df = self.avg_com_copy_df,
                                      parameter_dict = self.param_dict,
                                      loc_data_1 = snp_ar1,
                                      loc_data_2 = snp_ar2,
                                      time_data_1 = time_data_1,
                                      time_data_2 = time_data_2,
                                      data_type = data_type,
                                      file_name = 'COM_Traj_Vault_Depl',
                                      output_directory = new_output_dir)


    def save_data(self,file_name):
        """
        This function saves the average ensemble center of mass dataframe to the
        output directory.
        """
        self.avg_com_y_displace_df.to_csv(os.path.join(self.output_dir,'{}.csv'.format(file_name)))


#%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_directory",
                        help="Specify the path to the directory that contains this script and all other relevant scripts",
                    type = str)
    parser.add_argument("output_directory",
                        help="Specify the parent directory of the Migration Data",
                    type = str)
    parser.add_argument("flow_type",
                        help="Specify which type of flow you want to do the analysis on",
                    type = str)
    parser.add_argument("--input_directory",'-input_dir',nargs = '+',
                        help="Specify the parent directory of the Migration Data",
                    type = str)
    parser.add_argument("--input_snapshot_vault_directories",'-vault_snp_dirs',nargs = 2,
                        help = "Specify the path to the directories where the filament data corresponding to a pole vaulting motion resides in (2 must be specified)",
                        type = str)
    parser.add_argument("--preliminary_assessment",'-prelim',
                        help = "This argument specifies if you want to do a preliminary analysis of the data or not",
                        action = 'store_true',default = False)
    parser.add_argument("--find_fit_data",'-ffd',
                        help = "This argument specifies if you want to do do a fitting on the ensemble average y-center of mass trajectory",
                        action = 'store_true',default = False)
    
    ### Poiseuille flow data ###
    # args = parser.parse_args(['C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Scripts//02_COM_Trajectory_Data//',
    #                           'C://Users//super/OneDrive - University of California, Davis//Research/00_Projects//02_Shear_Migration//00_Scripts//02_COM_Trajectory_Data//02_Actual_Results//Poiseuille_Flow_Walls//COM_Data_0p25//',
    #                           "Poiseuille",'--input_directory','C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Remote_Data//Poiseuille_Flow_Walls//',
    #                           '--preliminary_assessment'])
    # args = parser.parse_args(['C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Scripts//02_COM_Trajectory_Data//',
    #                           'C://Users//super/OneDrive - University of California, Davis//Research/00_Projects//02_Shear_Migration//00_Scripts//02_COM_Trajectory_Data//02_Actual_Results//Poiseuille_Flow_Walls//COM_Data//',
    #                           "Poiseuille",'--input_directory','C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Remote_Data//Poiseuille_Flow_Walls//',
    #                           '--input_snapshot_vault_directories',
    #                           "C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Remote_Data//Poiseuille_Flow_Walls//VD_0p45//K_constant_UC_1p00//MB_10000//R_4//",
    #                           "C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Remote_Data//Poiseuille_Flow_Walls//VD_0p45//K_constant_UC_1p00//MB_500000//R_9//"])
    args = parser.parse_args(['C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Scripts//02_COM_Trajectory_Data//',
                              'C://Users//super/OneDrive - University of California, Davis//Research/00_Projects//02_Shear_Migration//00_Scripts//02_COM_Trajectory_Data//02_Actual_Results//Kolmogorov_Flows//COM_Data//',
                              "Kolmogorov",'--input_directory','C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Remote_Data//Kolmgorov_Flows_tf_0p20//'])
    os.chdir(args.project_directory)
    from calculations.adjust_position_data import adjust_position_data
    
    # args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s|%(filename)s|%(levelname)s|%(message)s',
            datefmt="%A, %B %d at %I:%M:%S %p")
    ### Read in Data ###
    all_ensbl_dat = all_ensemble_data(args.output_directory)
    if args.input_directory:
        for input_dir in args.input_directory:
            for root,dirs,files in os.walk(input_dir):
                for subdir_ in dirs:
                    check_file = os.path.join(root,subdir_,'filament_allstate.npy')
                    check_file2 = os.path.join(root,subdir_,'parameter_values.csv')
                    if os.path.exists(check_file) and os.path.exists(check_file2):
                        load_check_file2 = pd.read_csv(check_file2,index_col = 0,header = 0)
                        if "R" in subdir_ and float(load_check_file2.loc['Channel Upper Height','Value']) == 0.50:
                            match = re.search(r"R_(\d{1,})",subdir_)
                            if match and int(match.group(1)) <= 10:
                                
                                path_to_dir = os.path.join(root,subdir_)
                                replicate_number = int(match.group(1))
                                ensmbl_dat = indiv_ensemble_data(path_to_dir,replicate_number)
                                ensmbl_dat.find_ensemble_parameters()
                                
                                
                                ### Positional and Stress Data ###
                                all_ensbl_dat.add_position_stress_data(ensmbl_dat)
    
    """ Process Positional and Stress Data """
    all_ensbl_dat.compile_position_stress_data()
    all_ensbl_dat.calculate_average_com_position()
    all_ensbl_dat.filter_data(args.flow_type)
    all_ensbl_dat.calculate_avg_com_depletion_layer()
    all_ensbl_dat.calculate_ensemble_depletion_layer()
    
    """Fit COM Trajectory Data to Exponential Decay Equation"""
    if args.find_fit_data:
        all_ensbl_dat.fit_drift_velocity_exp_eqn()
        all_ensbl_dat.apply_height_class()
        
        ### Actual COM Data vs. Fitted Data ###
        all_ensbl_dat.ensemble_avg_ycom_traj_fits()
        
        ### COM Data Fit Parameters ###
        all_ensbl_dat.plot_avg_com_fits_params()
        
        ### COM Data Fit STDEV Parameters ###
        all_ensbl_dat.com_time_fits_params_stdev_plot()


    """ Preliminary assessment of data via plotting """
    # if args.preliminary_assessment:
        
        ### All ensemble Data ###
        # all_ensbl_dat.center_of_mass_ensemble_plot()
        # all_ensbl_dat.true_center_ensemble_plot()
    
        ### Average Ensemble Data ###
        # all_ensbl_dat.avg_center_of_mass_ensemble_plot()
        # all_ensbl_dat.avg_true_center_ensemble_plot()
    
        ### Depletion Layer ###
        # all_ensbl_dat.depl_layer_thickness_plot(data_type = 'average_com')
        # all_ensbl_dat.depl_layer_thickness_plot(data_type = 'ensemble_com')
        
    if args.flow_type == 'Kolmogorov':
        print("Done")
        # all_ensbl_dat.ensemble_com_distr_k_flows()
    
    elif args.flow_type == 'Poiseuille':
        mb_1 = 1e4
        mb_2 = 5e5
        
        ### Scale Depletion Layer Data ###
        all_ensbl_dat.fit_trajectory_depl_layer()
        
        
        ### Average COM Distribution-Subplots ###
        # all_ensbl_dat.ensemble_com_distr_poiseuille(mu_bar1 = mb_1,
                                                        # mu_bar2 = mb_2)
                
        if args.input_snapshot_vault_directories:
            position_data_1 = np.load(os.path.join(args.input_snapshot_vault_directories[0],'filament_allstate.npy'))
            position_data_2 = np.load(os.path.join(args.input_snapshot_vault_directories[1],'filament_allstate.npy'))
            
            time_data_1 = np.load(os.path.join(args.input_snapshot_vault_directories[0],'filament_time_vals_all.npy'))
            time_data_2 = np.load(os.path.join(args.input_snapshot_vault_directories[1],'filament_time_vals_all.npy'))
            
            adj_position_data_1 = adjust_position_data(position_data = position_data_1,
                                                        adj_centering = True,
                                                        adj_translation = False,
                                                        transl_val = 0)
            
            adj_position_data_2 = adjust_position_data(position_data = position_data_2,
                                                        adj_centering = True,
                                                        adj_translation = False,
                                                        transl_val = 0)
            
            all_ensbl_dat.ycom_traj_vault_snp_depl_layer(data_type = 'average_com',
                                                                      mu_bar1 = mb_1,
                                                                      mu_bar2 = mb_2,
                                                                      snp_ar1 = adj_position_data_1,
                                                                      snp_ar2 = adj_position_data_2,
                                                                      time_data_1 = time_data_1,
                                                                      time_data_2 = time_data_2)
    

    
    
    """Save data"""
    # all_ensbl_dat.save_data("All_Poi_Data")
    

#%% Sample figure to show fiber crossing centerline

mu_bar = 1e5
y0_int = 0.15

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})


all_df = all_ensbl_dat.all_ensemble_loc_data_df.copy()
all_df = all_df[(all_df['Mu_bar'] == mu_bar) & (all_df['Starting Vertical Displacement'] == y0_int)]
avg_df = all_ensbl_dat.avg_com_y_displace_df.copy()
avg_df = avg_df[avg_df['Mu_bar'] == mu_bar]
fig,axes = plt.subplots(figsize = (10,7),ncols = 2,layout = 'constrained')

sns.lineplot(x = 'Brownian Time',y = 'Center of Mass-y',data = all_df,
             hue = 'Rep Number',palette = 'tab10',legend = False,ax = axes[0])
sns.lineplot(x = 'Brownian Time',y = 'Center of Mass-y',data = avg_df,
             hue = 'Starting Vertical Displacement',palette = 'tab10',
             legend = False,ax = axes[1])

for n,ax in enumerate(axes):
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-2,-2))
    ax.xaxis.offsetText.set_fontsize(0)
    ax.set_xlim(-1e-6,5.01e-2)
    ax.set_xticks(np.linspace(0,5e-2,6))
    ax.set_ylim(-0.3,0.3)
    ax.set_yticks(np.linspace(-0.25,0.25,5))
    ax.set_xlabel(r"$t^{\text{Br}} \times 10^{-2}$",fontsize = 13,labelpad = 13)
    if n == 0:
        ax.set_ylabel(r"$y^{\text{com}} (t)$",fontsize = 13,labelpad = 5)
    elif n == 1:
        ax.set_ylabel(r"$\langle y^{\text{com}} (t) \rangle $",fontsize = 13,labelpad = 5)
    ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    ax.axhline(y = 0.25,xmin = 0,
                  xmax = 1,color = 'gray',alpha = 0.4,
                  linestyle = 'dashed')
    ax.axhline(y = -0.25,xmin = 0,
                  xmax = 1,color = 'gray',alpha = 0.4,
                  linestyle = 'dashed')
axes[0].text(x = 0.10e-2,y = 0.32,s = r"\textbf{(a)}",fontsize = 15)
axes[1].text(x = 0.10e-2,y = 0.32,s = r"\textbf{(b)}",fontsize = 15)
plt.savefig(os.path.join(all_ensbl_dat.output_dir,'fiber_confined_results.pdf'),dpi = 600,
            bbox_inches = 'tight',format = 'pdf')
plt.show()