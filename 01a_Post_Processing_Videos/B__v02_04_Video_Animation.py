# -*- coding: utf-8 -*-
"""
FILE NAME:      Video_Animation.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        B__v00_00_Plot_Filament_Length.py; 

DESCRIPTION:    This script will read the simulation data for a particular instance.
                It will also determine instances when the filament is bending 
                with the flow and assuming the filament bends into a J-shape,
                it will calculate the radii for those instances.

INPUT
FILES(S):       1) .NPY file that contains all positions of each discretized
                position of the filament for each time step during
                the duration of the simulation.
                2) .NPY file that contains all tension values of each discretized
                position of the filament for each time step for 
                the duration of the simulation.
                3) .NPY file that contains averaged extra particle stress tensor
                component values for each time step for 
                the duration of the simulation.
                4) .NPY file that contains filament length at each time step
                for each time step for the duration of the simulation.
                5) .NPY file that contains averaged elastic energy across the 
                filament for each time step for the duration of the simulation.
                6) .NPY file that contains all time steps used for the simulation.
                7) .CSV file that lists all parameters used for the run. 

OUTPUT
FILES(S):       1) .PNG file that plots the filament rigidity as a function of 
                s. 
                2) .PNG file that plots the filament length over the course of
                the simulation. 
                3) .PNG file that plots the elastic energy over the course of
                the simulation. 
                4) .PNG file that plots the filament stresses over the course of
                the simulation. 
                5) .PNG file that plots changes in filament's center of mass
                and true center over course of simulation.
                5) .MP4 file that animates the filament movement over the course of
                the simulation. 
                6) .MP4 file that animates the filament tension over the course of 
                the simulation. 
                7) .MP4 file that animates the filament position and tension 
                simultaneously over the course of the simulation.
                8) .MP4 file that animates the filament position and center of 
                mass simultaneously over the course of the simulation.
                9) .MP4 file that animates the filament position and true center
                simultaneously over the course of the simulation.
                10) .PNG files that snapshots the filament's position at various 
                timepoints. 
                11) .PNG files that snapshots the filament's tension at various 
                timepoints.         


INPUT
ARGUMENT(S):    None

CREATED:        08Mar23

MODIFICATIONS
LOG:
17Aug21         Cleaned up code.
17Aug21         Reformatted code to adjust for time scaling in saving .NPY files.
06Aug21         Reformatted code to run multiple directories (Brownian replicates).
                Code now displays information about L/lp for Brownian
15Oct21         Reformatted code to multiprocess on all CPU cores. 
18Oct21         Reformatted code to plot filament center of mass, true center 
                data if rotation data is present. 
04Mar23         Code can now determine instances of bending similar to J-shapes 
                during the simulation; during these instances, the radius of
                the filament is calculated.
            

LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.9.13

VERSION:        2.1

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:     Add code to compile all center of mass, true center, stress data
                across all simulations.

NOTE(S):        N/A

"""
import re, sys, os, math, logging, time, argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import special
from scipy.signal import savgol_filter, find_peaks, peak_widths, peak_prominences

### comment this line if running from terminal ###
current_dir = os.chdir('C:\\Users\\super\\OneDrive - University of California, Davis\\Research\\00_Projects\\02_Shear_Migration\\00_Scripts\\01a_Post_Processing_Videos\\')
sys.path.append(current_dir)

### Import other files & plotting features ###
from calculations import first_derivative,second_derivative
from misc.create_dir import create_dir
from plot.Plot_Filament_Length import plot_fil_length
from plot.Plot_Filament_EE_Length import plot_fil_ee_length
from plot.Plot_Filament_Elastic_Energy import plot_fil_elastic_energy
from plot.Plot_Filament_Stresses import plot_fil_stresses
from animate import video_animation,new_frame_double_plot ,new_frame_single_plot

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.sans-serif": "Times",
#     'text.latex.preamble': r'\usepackage{amsmath}'})

#%%

class filament_data():
    slicing_factor = 1
    
    def __init__(self, target_dir,params_df,rotate_df,length_ar,
                 position_ar,tension_ar,stress_ar,elastic_ar,time_vals_ar):
        """
        Initialization of the class. 
        """
        self.params_df = pd.read_csv(params_df,index_col = 0,header = 0)
        ### Constants ###
        self.output_dir = target_dir
        self.L = float(self.params_df.loc['Filament Length','Value'])
        try:
            self.lp = float(self.params_df.loc['Filament Persistence Length','Value'])
            self.llp = self.L/self.lp
        except ValueError: #For non-Brownian simulations
            self.lp = None
            self.llp = None
        self.N = int( self.params_df.loc['N','Value'])
        self.ds = 1/(self.N-1)
        self.dt = float(self.params_df.loc['Array Time Step Size','Value'])
        self.s = np.linspace(float( self.params_df.loc['Filament s start','Value']),
                             float(self.params_df.loc['Filament s end','Value']),
                             self.N,dtype = float)
        self.true_center_loc = np.where(self.s == 0)[0][0]
        self.c = float(self.params_df.loc['c','Value'])
        self.mu_bar = float(self.params_df.loc['Mu_bar','Value'])
        self.top = int(self.params_df.loc['theta_num','Value'])
        self.bottom = int(self.params_df.loc['theta_den','Value'])
        self.rigidity_suffix = self.params_df.loc['Rigidity Function Type','Value']
        self.channel_height = np.abs(float(self.params_df.loc['Channel Upper Height','Value']))
        self.U_centerline = float(self.params_df.loc['Poiseuille U Centerline','Value'])
        self.kflow_freq = float(self.params_df.loc['Kolmogorov Frequency','Value'])
        if self.params_df.loc['Kolmogorov Phase','Value'] == 'Pi':
            self.kflow_phase_text = 'Pi'
            self.kflow_phase_val = np.pi
        else:
            self.kflow_phase_text = '{:.2f}'.format(float(self.params_df.loc['Kolmogorov Phase','Value']))
            self.kflow_phase_val = float(self.params_df.loc['Kolmogorov Phase','Value'])
        
        
        self.time_values = time_vals_ar
        self.time_start_val = 0
        
        ### Select all time values ###
        self.time_end_val = np.where(time_vals_ar>=self.time_values[-1])[0][0] + 1
        
        ### Use this method if you don't want to use all time values ###
        # self.time_end_val = np.where(time_vals_ar>=35)[0][0] + 1
        
        
        self.time_values = time_vals_ar[self.time_start_val:self.time_end_val]

        if self.time_values[-1] in self.time_values[::self.slicing_factor]:
            self.total_frames = np.r_[:len(self.time_values):self.slicing_factor]
        else:
            self.total_frames = np.append(np.r_[:len(self.time_values):self.slicing_factor],len(self.time_values)-1)
        
        ##########
        
        self.total_frames = np.r_[:len(self.time_values):self.slicing_factor]
        self.center = np.where(self.s == 0)[0][0]
        
                
        ### Rigidity Profiles ###
        
        if self.rigidity_suffix == 'K_constant':
            self.K = np.ones(self.s.shape[0],dtype = float)
            self.Ks = np.zeros(self.s.shape[0],dtype = float)
            self.Kss = np.zeros(self.s.shape[0],dtype = float)
            self.rigidity_title = "$\kappa(s) = 1$"
        elif self.rigidity_suffix == 'K_parabola_center_l_stiff':
            self.K = 1/2 + 2*self.s**2
            self.Ks = 4*self.s
            self.Kss = 4*np.ones(self.s.shape[0],dtype = float)
            self.rigidity_title = "$\kappa(s) = \frac{1}{2} + 2s^{2}} $"
        elif self.rigidity_suffix == 'K_parabola_center_m_stiff':
            self.K = 1.5 - 2*(self.s**2)
            self.Ks = -4*self.s
            self.Kss = -4*np.ones(self.s.shape[0],dtype = float)
            self.rigidity_title = "$\kappa(s) = \frac{3}{2} - 2s^{2} $"
        elif self.rigidity_suffix == 'K_linear':
            self.K = self.s+1
            self.Ks = 1*np.ones(self.s.shape[0],dtype = float)
            self.Kss = 0*np.ones(self.s.shape[0],dtype = float)
            self.rigidity_title = "$\kappa(s) = s+1 $"            
        elif self.rigidity_suffix == 'K_dirac_center_l_stiff':
            self.K = 1-0.5*np.exp(-100*self.s**2)
            self.Ks = 100*self.s*np.exp(-100*self.s**2)
            self.Kss = np.exp(-100*self.s**2)*(100-2e4*self.s**2)
            self.rigidity_title = "$\kappa(s) = 1- \frac{1}{2} e^{-100s^{2}} $"            
        elif self.rigidity_suffix == 'K_dirac_center_l_stiff2':
            self.K = 1-0.5*np.exp(-500*self.s**2)
            self.Ks = 500*self.s*np.exp(-500*self.s**2)
            self.Kss = np.exp(-500*self.s**2)*(500-5e5*self.s**2)
            self.rigidity_title = "$\kappa(s) = 1- \frac{1}{2} e^{-500s^{2}} $"
        elif self.rigidity_suffix == 'K_dirac_center_m_stiff':
            self.K = 1+np.exp(-100*self.s**2)
            self.Ks = -200*self.s*np.exp(-100*self.s**2)
            self.Kss = 200*np.exp(-100*self.s**2)*(200*self.s**2-1)
            self.rigidity_title = "$\kappa(s) = 1 + e^{-100s^{2}} $"
        elif self.rigidity_suffix == 'K_parabola_shifted':
            self.K = 1.5-0.5*(self.s-0.5)**2
            self.Ks = -1*self.s-0.5
            self.Kss = -1*np.ones(self.s.shape[0],dtype = float)
            self.rigidity_title = r"$\kappa(s) = \frac{3}{2}-\frac{1}{2}\left(s-\frac{1}{2}\right)^{2} $"
        elif self.rigidity_suffix == 'K_error_function':
            self.K = special.erf(10*self.s)+2
            self.Ks = (20/np.sqrt(np.pi))*np.exp(-100*self.s**2)
            self.Kss = (-4000*self.s/np.sqrt(np.pi))*np.exp(-100*self.s**2)
            self.rigidity_title = "$\kappa(s) = 2 + erf(10s) $"
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff':
            self.K = 1-0.5*np.exp(-100*(self.s+0.25)**2)
            self.Ks = 100*(self.s+0.25)*np.exp(-100*(self.s+0.25)**2)
            self.Kss = np.exp(-100*(self.s+0.25)**2)*-2e4*(self.s**2+0.5*self.s+0.0575)
            self.rigidity_title = r"$\kappa(s) = 1- \frac{1}{2} e^{-100\left(s + \frac{1}{4}\right)^{2}} $" 
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff2':
            self.K = 1-0.5*np.exp(-500*(self.s+0.25)**2)
            self.Ks = 500*(self.s+0.25)*np.exp(-500*(self.s+0.25)**2)
            self.Kss = np.exp(-500*(self.s+0.25)**2)*-5e5*(self.s**2+0.5*self.s+0.0615)
            self.rigidity_title = "$\kappa(s) = 1- \frac{1}{2} e^{-500\left(s + \frac{1}{4}\right)^{2}} $" 
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff3':
            self.K = 1-0.5*np.exp(-1000*(self.s+0.25)**2)
            self.Ks = 1000*(self.s+0.25)*np.exp(-1000*(self.s+0.25)**2)
            self.Kss = np.exp(-1000*(self.s+0.25)**2)*-2e6*(self.s**2+0.5*self.s+0.062)
            self.rigidity_title = "$\kappa(s) = 1- \frac{1}{2} e^{-1000\left(s + \frac{1}{4}\right)^{2}} $" 
            
        ##### Load Raw Filament Data & Filter on designated end time #####
        self.rotate_data = rotate_df
        self.length_data = length_ar[self.time_start_val:self.time_end_val]
        self.position_data = position_ar[:,:,self.time_start_val:self.time_end_val]
        self.position_data[:,0,:] = self.position_data[:,0,:] - self.position_data[:,0,:].mean(axis = 0)#Adjust for any translation in x-coordinates
        self.tension_data = tension_ar[:,self.time_start_val:self.time_end_val]
        self.stress_data = stress_ar[:,:,self.time_start_val:self.time_end_val]
        self.elastic_data = elastic_ar[self.time_start_val:self.time_end_val]
        self.indiv_true_center = self.position_data[self.center,:,:]
        self.center_mass = self.position_data.mean(axis = 0).T #Rows are each time point, columns are x,y,z, components
        self.true_center = self.position_data[self.true_center_loc,:,:].T #Rows are time points, columns are x,y,z components
        
        #### Transformed Data ####
        self.N1_stress = self.stress_data[0,0,:] - self.stress_data[1,1,:]
        self.N2_stress = self.stress_data[1,1,:] - self.stress_data[2,2,:]
        self.sig12_stress = self.stress_data[0,1,:]
        self.sig21_stress = self.stress_data[1,0,:]
        

    def calculate_curvature(self):
        """
        This function calculates the curvature along the filament according to
        the following equation: 
        https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization.
        """
        
        ### Calculate first and second derivatives ###
        self.xs = first_derivative.first_derivative(base_array = self.position_data,
                              deriv_size = self.ds,axis = 0,
                              dim = 3)
        self.xss = second_derivative.second_Derivative(base_array = self.position_data,
                                deriv_size = self.ds,)
        
        ## Always positive curvature ##
        self.k_curvature = (np.abs((self.xs[:,0,:]*self.xss[:,1,:]) - (self.xs[:,1,:]*self.xss[:,0,:])))/\
            (((self.xs[:,0,:]**2)+(self.xs[:,1,:]**2))**(1.5))
            
        ## Curvature depending on filament orientation & orientation of s ##
        # self.k_curvature = (((self.xs[:,0,:]*self.xss[:,1,:]) - (self.xs[:,1,:]*self.xss[:,0,:])))/\
            # (((self.xs[:,0,:]**2)+(self.xs[:,1,:]**2))**(1.5))
        self.k_curvature = np.nan_to_num(self.k_curvature,copy = False)
    
    def calculate_angle_orientation(self):
        self.all_angles = np.mod(np.arctan2(self.xs[:,1,:],self.xs[:,0,:]),2*np.pi)
        self.filament_angles = np.rad2deg(self.all_angles)

    def detect_elastic_peaks(self):
        """
        This function determines instances of filament bending & rotation in 
        flow based on a pre-set criteria. The identification of this instances
        is based on the elastic energy of the filament peaking. This function
        also calculates the 99% of the peak width as well
        """
        
        ### Scale parameters based on mu_bar value ###
        self.threshold = self.mu_bar**0.22
        self.min_elastic_distance = int(np.ceil((self.mu_bar**-0.80)/self.dt))
        self.elastic_peak_idx,_ = find_peaks(self.elastic_data,height = self.threshold,
                                             prominence = 0.95*self.threshold,
                                             distance =self.min_elastic_distance0)
        self.peak_widths_results = peak_widths(self.elastic_data,
                                               self.elastic_peak_idx,rel_height = 0.99) #Find peak start and end based on 99% of height value
        self.peak_width = self.peak_widths_results[1]
        self.elastic_peak_duration = self.peak_widths_results[0]*self.dt
        self.peak_start_idx,self.peak_end_idx = self.peak_widths_results[2:]
        
    def find_time_vals_interest(self):
        """
        This function finds the time values at which there's instances of filament
        bending.
        """
        self.snapshot_points = self.time_values[self.elastic_peak_idx]
        

    def adjust_elastic_peak_width(self):
        """
        This function determines where the start and end coordinates of the elastic
        peak.
        """
        
                
        ### Adjust width of peak by specified time ###
        self.time_factor = int(2e-5/self.dt)
        self.peak_start_idx = np.array([math.floor(i/10)*10-self.time_factor for \
                                        i in self.peak_start_idx])
        self.peak_end_idx = np.array([math.ceil(i/10)*10+self.time_factor for \
                                        i in self.peak_end_idx])
            
            
        #Check to see if peak_intervals don't overlap with each other
        self.adj_elastic_peak_idx = self.elastic_peak_idx.copy()
        self.adj_elastic_peak_duration = self.elastic_peak_duration.copy()
        self.adj_peak_start_idx = self.peak_start_idx.copy()
        self.adj_peak_end_idx = self.peak_end_idx.copy()
        
        ### Remove peak if end point they extend past array size and filament is considered bending ###
        if self.peak_end_idx[-1] > self.time_data.shape[0]-1 and self.elastic_data[-1] >= 0.5:  
            self.end_array = np.array([-1])
        elif self.peak_end_idx[-1] > self.time_data.shape[0]-1 and self.elastic_data[-1] < 0.5:
            self.peak_end_idx[-1] = self.time_data.shape[0]-1
            self.end_array = ([])
        else:
            self.end_array = ([])
        
        ### Fix starting point of first peak if it is outside time domain ###
        if self.peak_start_idx[0] < 0:
            self.peak_start_idx[0] = 0
        self.prev_peak_start_idx = self.peak_start_idx[:-1].copy()
        self.prev_peak_end_idx = self.peak_end_idx[:-1].copy()
        self.overlap = (self.peak_start_idx[1:] < self.prev_peak_end_idx) & (self.peak_start_idx[1:] > self.prev_peak_start_idx)
        self.indices = np.where(self.overlap)[0]
        self.indices_to_delete = np.unique(np.concatenate([self.indices,self.indices+1,self.end_array])).astype(int)
        if self.indices_to_delete.shape[0] > 0:
            self.adj_elastic_peak_idx = np.delete(self.elastic_peak_idx,self.indices_to_delete)
            self.adj_elastic_peak_duration = np.delete(self.elastic_peak_duration,self.indices_to_delete)
            self.adj_peak_start_idx = np.delete(self.peak_start_idx,self.indices_to_delete)
            self.adj_peak_end_idx = np.delete(self.peak_end_idx,self.indices_to_delete)
        else:
            self.adj_elastic_peak_idx = self.elastic_peak_idx.copy()
            self.adj_elastic_peak_duration = self.elastic_peak_duration.copy()
            self.adj_peak_start_idx = self.peak_start_idx.copy()
            self.adj_peak_end_idx = self.peak_end_idx.copy()

        logging.info("Finished identifying elastic energy peaks.")
        
    def find_max_curvature(self):
        """
        This function finds where on the filament the max curvature is located at.
        """
        self.curvature_threshold = 8 #Minimum value or curvature to be considered for
        self.curvature_max_dist = int(np.ceil(self.N/15)) # Minimum number of points along filament between curvature calculations; set for 0.15L along filament 
        self.curv_loc_factor = 0.35 #Max curvature value should exist within 0.35L of either side of the filament
        self.max_curvature_data = []
        self.err_curvature_data = []
        
        for self.ee_idx in self.elastic_peak_idx:
            self.curv_peak_idx,_ = find_peaks(self.k_curvature[:,self.ee_idx],
                                              height = self.curvature_threshold,
                                              distance = self.curvature_max_dist) #When elastic energy peaks, find peak curvature
            if self.curv_peak_idx.shape[0] == 1:
                self.max_curv_val = self.k_curvature[self.curv_peak_idx[0],self.ee_idx]
                if self.curv_peak_idx[0]  >= 0.35*self.N and self.curv_peak_idx[0] <= 0.65*self.N:
                    logging.warning(
                        "The location of max curvature is too close to the center and not a J-shape! s index of max curvature: {}. Results will be appended to the separate list".format(
                            self.curv_peak_idx[0]))
                    self.curvature_results = {"Elastic Energy Time Index": [self.ee_idx],
                                         "Curvature Peak Filament Index": int(self.curv_peak_idx),
                                         "Max Curvature Value": self.max_curv_val}
                    self.err_curvature_data.append(self.curvature_results)
                else:
                    self.curvature_results = {"Elastic Energy Time Index": [self.ee_idx],
                                         "Curvature Peak Filament Index": int(self.curv_peak_idx),
                                         "Max Curvature Value": self.max_curv_val}
                    self.max_curvature_data.append(self.curvature_results)
            else:
                logging.warning("Error! There are two peaks at t = {}. Curvature data will be written to the separate list.".format(self.time_data[self.ee_idx]))
                for self.max_curv_val_idx in self.curv_peak_idx.shape[0]:
                    self.current_curvature_val = self.k_curvature[self.max_curv_val_idx,self.ee_idx]
                    self.plot_f_pos_curvature_debug(self.curv_peak_idx,self.ee_idx)
                    self.curvature_results = {"Elastic Energy Time Index": [self.ee_idx],
                                         "Curvature Peak Filament Index": self.max_curv_val_idx,
                                         "Max Curvature Value": self.current_curvature_val}
                    self.err_curvature_data.append(self.curvature_results)
            
        
        #check if there's any instances of J-shape bending after reading through simulation data
        if self.max_curvature_data:
            self.curvature_vals_df = pd.concat(
                [pd.DataFrame.from_dict(i) for i in self.max_curvature_data],ignore_index = True)
            self.curvature_vals_df.set_index("Elastic Energy Time Index",inplace = True)
            self.curvature_vals_df.dropna(how = 'any',inplace = True)
            self.curvature_vals_df['Elastic Energy Time Index'] = self.curvature_vals_df.index.values.copy()
        else:
            self.curvature_vals_df = pd.DataFrame()
        
        if self.err_curvature_data:
            self.err_vals_df = pd.concat(
                [pd.DataFrame.from_dict(i) for i in self.err_curvature_data],ignore_index = True)
            self.err_vals_df.set_index("Elastic Energy Time Index",inplace = True)
            self.err_vals_df.dropna(how = 'any',inplace = True)
            self.err_vals_df['Elastic Energy Time Index'] = self.err_vals_df.index.values.copy()
        else:
            self.err_vals_df = pd.DataFrame()
        logging.info("Finished identifying the location of maximum curvature along the filament.")
        
    def init_compile(self):
        """
        This function initiates an empty list to store all of the dictionary data in.
        """
        self.all_curvature_data_list = []
        
        
    def add_radius_data(self):
        """
        This function reads in the DataFrame that specifies the location of 
        elastic energy peaks and 
        """
        self.init_compile()
        
        for i,ee_peak_idx in self.curvature_vals_df.index.values:
            
            start_com_y = self.center_mass[self.adj_peak_start_idx[i],1]
            end_com_y = self.center_mass[self.adj_peak_end_idx[i],1]
            start_tc_y = self.position_data[self.true_center_loc,1,self.adj_peak_start_idx[i]]
            end_tc_y = self.position_data[self.true_center_loc,1,self.adj_peak_end_idx[i]]
            peak_curv_loc = int(self.curvature_vals_df.loc[ee_peak_idx,'Curvature Peak Filament Index'])
            max_curv_val = self.curvature_vals_df.loc[ee_peak_idx,'Max Curvature Value']
            
            net_com_y = np.abs(end_com_y) - np.abs(start_com_y)
            net_tc_y = np.abs(end_tc_y)  - np.abs(start_tc_y)
            
            
            if np.isnan(peak_curv_loc):
                s_peak_curv_loc = np.array([np.nan])
            else:
                s_peak_curv_loc = np.array([self.s[peak_curv_loc]])
                
            if np.isnan(max_curv_val):
                radius_bending = np.nan
            else:
                radius_bending = max_curv_val**-1
                
            ### Find time points between elastic peaks ###
            if i != 0:
                self.curr_ee_peak_time = self.time_data[self.adj_elastic_peak_idx[i]]
                self.prev_ee_peak_time = self.time_data[self.adj_elastic_peak_idx[i-1]]
                ee_peak_time_diff = self.curr_ee_peak_time - self.prev_ee_peak_time
            else:
                ee_peak_time_diff = np.array([0])
            self.indiv_ensemble_curv_dict = {"Rigidity Suffix":self.rigidity_profile,
                                        "Mu_bar": self.mu_bar,
                                        "N": self.N,
                                        'Channel Height': self.channel_height,
                                        'Poiseuille U Centerline': self.U_centerline,
                                        'Kolmogorov Frequency': self.k_freq,
                                        'Kolmogorov Phase': self.k_phase,
                                        'Starting Vertical Displacement': self.vert_displ,
                                        'Steric Velocity Exponential Coefficient':self.velo_exp,
                                        'Steric Velocity Gap Criteria': self.velo_gap,
                                        'Flow Type': self.flow_type,
                                        'Brownian Use':self.brownian_use,
                                        'Sterics Use': self.sterics_use,
                                        "Max Elastic Energy Index Value": ee_peak_idx,
                                        "Time of Max Elastic Energy": self.time_data[ee_peak_idx],
                                        "Elastic Energy Value at Peak": self.elastic_data[ee_peak_idx],
                                        "Start Time of Elastic Energy Peak": self.time_data[self.adj_peak_start_idx[i]],
                                        "End Time of Elastic Energy Peak": self.time_data[self.adj_peak_end_idx[i]],
                                        "Start Time Index of Elastic Energy Peak": self.adj_peak_start_idx[i],
                                        "End Time Index of Elastic Energy Peak": self.adj_peak_end_idx[i],
                                        "Time Difference Between Current and Previous Rotation":ee_peak_time_diff,
                                        "Starting COM-y":start_com_y,
                                        "COM-y at Elastic Peak": self.center_mass[ee_peak_idx,1],
                                        "Ending COM-y": end_com_y,
                                        "Starting True Center-y":start_tc_y,
                                        "True Center-y at Elastic Peak":self.position_data[self.true_center_loc,1,ee_peak_idx],
                                        "Ending True Center-y":end_tc_y,
                                        "Net COM-y": net_com_y,
                                        "Net True Center-y": net_tc_y,
                                        "Rotation Duration": self.adj_elastic_peak_duration[i],
                                        "Drift Rotation Velocity": net_com_y/self.adj_elastic_peak_duration[i],
                                        "Peak Curvature Index Value": peak_curv_loc,
                                        "Location of Peak Curvature": s_peak_curv_loc,
                                        "Peak Curvature Value":max_curv_val,
                                        "Radius of Bending": radius_bending}
            self.all_ensemble_curvature_data_list.append(self.indiv_ensemble_curv_dict)
            
    def compile_position_curvature_data(self):
        """
        This function converts the dictionary of all ensemble curvature data 
        into a Pandas DataFrame.
        """
        self.all_ensemble_k_df = pd.concat(
            [pd.DataFrame.from_dict(i) for i in self.all_ensemble_curvature_data_list],ignore_index = True)
       
        
        ### Calculate Local Shear Rate ###
        exp_groups = self.all_ensemble_k_df.groupby(by = ['Flow Type'])
        for group in exp_groups.groups.keys():
            group_df = exp_groups.get_group(group)
            if group == 'Poiseuille':
                self.all_ensemble_k_df.loc[group_df.index.values,'Local Shear Rate'] = 2*group_df['Poiseuille U Centerline'] * np.abs(group_df['Starting COM-y'])/(group_df['Channel Height']**2)
            elif group == 'Shear':
                self.all_ensemble_k_df.loc[group_df.index.values,'Local Shear Rate'] = 1
            elif group == 'Kolmogorov':
                self.all_ensemble_k_df.loc[group_df.index.values,'Local Shear Rate'] = group_df['Kolmogorov Frequency'] * group_df['Kolmogorov Phase'] * np.cos(group_df['Kolmogorov Frequency']* group_df['Kolmogorov Phase'] * group_df['Starting COM-y'])
                
                
        self.all_ensemble_k_df['Displacement Type'] = self.all_ensemble_k_df['Net COM-y'].apply(lambda x: self.classify_displ_type(x))
        self.all_ensemble_k_df['ABS Net COM-y'] = np.abs(self.all_ensemble_k_df['Net COM-y'])
                
        ### Calculated Position-dependent Mu_bar ###
        self.all_ensemble_k_df['Mu_bar times Starting COM-y'] =  self.all_ensemble_k_df['Mu_bar'] *  self.all_ensemble_k_df['Starting COM-y']
        
    
    def classify_displ_type(self,net_displ_val):
        """
        This function classifies whether or not the filament flipped or or down based
        on the center of mass positions of the adjusted peak start and end values. 
        
        Inputs:
            
        net_displ_val:      Transverse distance the filament moved due to a flipping
                            event. 
        """
        if net_displ_val > 0:
            displ_type = "Up"
        elif net_displ_val < 0:
            displ_type = 'Down'
        else:
            displ_type = 'Stagnant'
        return displ_type
            
                    
        
#%%      
### Run Post-processing ###

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_directory",
                        help="Specify the parent directory of the Migration Data",
                    type = str)
    parser.add_argument("output_directory",
                        help="Specify the parent directory of the Migration Data",
                    type = str)
    parser.add_argument("--radius_calculations","-rad_calc",
                        help = 'Specify whether or not you want to calculate the radius of the filament bend during maximum elastic energy',
                        action = 'store_true')
    parser.add_argument("--plot_data","-pd",
                        help = 'Specify whether or not you want to plot results like length, elastic energy, stresses, etc.',
                       action = 'store_true')
    parser.add_argument("--animate_data","-ad",
                        help = 'Specify whether or not you want to plot results like length, elastic energy, stresses, etc.',
                        action = 'store_true')
    # args = parser.parse_args()
    
    #Uncomment this section when running it from an IDE unless argument based commands are configured
    args = parser.parse_args(['C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/02_Shear_Migration/00_Remote_Data/Poiseuille_Flow_Walls_Short_Video_Data/VD_0p45/K_constant_UC_1p00/MB_10000/R_4/',
                              'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/02_Shear_Migration/00_Remote_Data/Poiseuille_Flow_Walls_Short_Video_Data/VD_0p45/K_constant_UC_1p00/MB_10000/R_4/',
                              '--animate_data'])
    # args = parser.parse_args(['C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/02_Shear_Migration/00_Remote_Data/Poiseuille_Flow_Walls/VD_0p05/K_constant_UC_1p00/MB_50000/R_1/',
    #                           'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/02_Shear_Migration/00_Remote_Data/Poiseuille_Flow_Walls/VD_0p05/K_constant_UC_1p00/MB_50000/R_1/',
    #                           '--animate_data'])
    
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s|%(filename)s|%(levelname)s|%(message)s',
            datefmt="%A, %B %d at %I:%M:%S %p")
    
    
    tru_start_time = time.perf_counter()
    filament_data = filament_data(target_dir = args.output_directory,
                                  params_df = os.path.join(args.input_directory,'parameter_values.csv'),
        rotate_df = None,length_ar = np.load(os.path.join(args.input_directory,'filament_length.npy')),
         position_ar = np.load(os.path.join(args.input_directory,'filament_allstate.npy')),
         tension_ar = np.load(os.path.join(args.input_directory,'filament_tension.npy')),
         stress_ar = np.load(os.path.join(args.input_directory,'filament_stress_all.npy')),
         elastic_ar = np.load(os.path.join(args.input_directory,'filament_elastic_energy.npy')),
         time_vals_ar = np.load(os.path.join(args.input_directory,'filament_time_vals_all.npy')))
    
    ### Determine when the filament bends into a j-shape ###
    if args.radius_calculations:
        filament_data.calculate_curvature()
        filament_data.calculate_angle_orientation()
        filament_data.detect_elastic_peaks()
        filament_data.find_time_vals_interest()
        filament_data.adjust_elastic_peak_width()
        filament_data.find_max_curvature()
        filament_data.add_radius_data()
        filament_data.compile_position_curvature_data()
            
        
    
    
    
    ### Plotting Routines ###
    #if args.plot_data:
        # plot_fil_length(length_data = filament_data.length_data,
        #                 time_data = filament_data.time_values,
        #                 output_directory = args.output_directory)
        
        # plot_fil_ee_length(position_data = filament_data.position_data,
        #                    time_data = filament_data.time_values,
        #                    output_directory = args.output_directory)
        
        # plot_fil_elastic_energy(ee_data = filament_data.elastic_data,
        #                         time_data = filament_data.time_values,
        #                         ee_peak_idx = filament_data.curvature_vals_df['Elastic Energy Time Index'].values,
        #                         output_directory = args.output_directory)
        
        # plot_fil_stresses(stress_data = filament_data.stress_data,
        #                   time_data = filament_data.time_values,
        #                   output_directory = args.output_directory)
        
    if args.animate_data:
        video_animation.filament_position(filament_data = filament_data, 
                                          flow_type = 'Poiseuille',
                                          brownian = True,
                                          sterics = True ,
                                          upper_channel_height = filament_data.channel_height,
                                          lower_channel_height = -filament_data.channel_height)
        
        # video_animation.filament_position_tension(filament_data = filament_data, 
        #                                   flow_type = 'Poiseuille',
        #                                   brownian = True,
        #                                   sterics = True,
        #                                   upper_channel_height = filament_data.channel_height,
        #                                   lower_channel_height = -filament_data.channel_height)
    tru_end_time = time.perf_counter()
    logging.info("Finished all calculations and plotting routines.Time to completion: {:.0f} seconds.".format(tru_end_time - tru_start_time))
    

