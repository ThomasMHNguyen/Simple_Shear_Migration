# -*- coding: utf-8 -*-
"""
FILE NAME:      Process_Filament_Flip_Data.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    This script will read in all of the simulation data files regarding
                shear-induced migration in either shear or Poiseuille flow and process
                it to calculate the radius of bending of J-shapes and net displacement 
                due to the flipping events. 

INPUT
FILES(S):       1) .NPY file that contains all positions of each discretized
                position of the filament for the duration of the simulation.
                3) .CSV file that lists all parameters used for run. 

OUTPUT
FILES(S):       1) .PNG file that shows the trajectory of each ensemble simulation
                at a given flow strength value, starting displacement, and rigidity
                type.
                2) .PNG file that shows the average ensemble simulation trajectory 
                at a given flow strength value, starting displacement, and rigidity
                type.
                3) .PNG file that shows the average ensemble simulation N1 stress
                difference value at a given flow strength value, starting displacement,
                and rigidity type.
                4) .PNG file that shows the average ensemble simulation N2 stress
                difference value at a given flow strength value, starting displacement,
                and rigidity type.
                5) .PNG file that shows the average ensemble simulation S12 stress
                value at a given flow strength value, starting displacement,
                and rigidity type.
                6) .PNG file that shows the average ensemble simulation drift velocity
                at a given flow strength value, starting displacement, and rigidity type.
                7) .PNG filethat shows the relationship between average net displacement
                and flow strength based on a centerline velocity value. 


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
                A__v01_03_Process_Filament_Flip_Data.py. Simplified code by 
                removing features to identify instances of bending and 
                calculating curvature. These instances will be part of another script. 
22Nov22         6) Migrated Plotting functions to a separate code. 

    
            
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.8.8

VERSION:        1.3

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:     

NOTE(S):        N/A

"""
import re, sys, os, argparse, math, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.signal import find_peaks,peak_widths
from scipy.optimize import curve_fit

### Use LaTeX to generate plots ###
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})
#%%
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
        self.stress_data = np.load(os.path.join(self.dir_,'filament_stress_all.npy'))
        self.elastic_data = np.load(os.path.join(self.dir_,'filament_elastic_energy.npy'))
        self.center_mass = center_of_mass(position_data = self.position_data,
                                          position = 0, dim = 3,
                                          adj_centering = False,
                                          adj_translation = False, 
                                          transl_val = 0)
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
        self.s = np.linspace(float(self.params_df.loc['Filament s start','Value']),
                                   float(self.params_df.loc['Filament s end','Value']),
                                   self.N)
        self.true_center_loc = np.where(self.s == 0)[0][0]
        self.flow_type = self.params_df.loc['Flow Type','Value']
        self.sterics_use = self.params_df.loc['Sterics Use','Value']
        self.brownian_use = self.params_df.loc['Brownian Use','Value']
        self.kflow_freq = float(self.params_df.loc['Kolmogorov Frequency','Value'])
        if self.params_df.loc['Kolmogorov Phase','Value'] == 'Pi':
            self.kflow_phase_text = 'Pi'
            self.kflow_phase_val = np.pi
        else:
            self.kflow_phase_text = '{:.2f}'.format(float(self.params_df.loc['Kolmogorov Phase','Value']))
            self.kflow_phase_val = float(self.params_df.loc['Kolmogorov Phase','Value'])
        
                   
    
    def calculate_curvature(self):
        """
        This function calculates the curvature along every point along the filament.
        """
        self.xs = first_derivative(base_array = self.position_data,
                                   deriv_size = self.ds, axis = 0,
                                   ar_size = self.N,
                                   dim = 3)
        self.xss = second_derivative(base_array = self.position_data,
                                     deriv_size = self.ds, axis = 0,
                                     ar_size = self.N,
                                     dim = 3)
         
        ## Always positive curvature ##
        self.k_curvature = (np.abs((self.xs[:,0,:]*self.xss[:,1,:]) - (self.xs[:,1,:]*self.xss[:,0,:])))/\
            (((self.xs[:,0,:]**2)+(self.xs[:,1,:]**2))**(1.5))

            
        self.k_curvature = np.nan_to_num(self.k_curvature,copy = False)
        logging.info("Finished with calculating curvature along the filament.")
        
    def detect_elastic_peaks(self):
        """
        This function finds the location where the elastic energy of the filament
        peaks and adjusts the start and end location of the peaks.
        """
        
        
        ### 1-size fits all parameters ###
        # self.threshold = 10 #Minimum value of elastic energy to be considered
        # self.min_elastic_distance = int(1e-5/self.dt) #peaks should be at least 1e-4 in time away from each other 
        # self.elastic_peak_idx,_ = find_peaks(self.elastic_data,height = 10,
                                              # distance = 100)
        
        ### Scale parameters based on mu_bar value ###
        self.threshold = self.mu_bar**0.22
        self.min_elastic_distance = int(np.ceil((self.mu_bar**-0.80)/self.dt))
        
        
        self.elastic_peak_idx,_ = find_peaks(self.elastic_data,height = self.threshold,prominence = 0.95*self.threshold,
                                              distance = self.min_elastic_distance)
        
        
        self.peak_widths_results = peak_widths(self.elastic_data,self.elastic_peak_idx,rel_height = 0.99) #Find peak start and end based on 99% of height value
        self.elastic_peak_duration = self.peak_widths_results[0]*self.dt
        self.peak_start_idx,self.peak_end_idx = self.peak_widths_results[2:]
        
        
        if self.elastic_peak_idx.size > 0:
            ##### Widen peak start and end to make sure that center of mass has stabilized #####
        
            ### Widen time of peak by specified time ###
            # self.time_factor = int(2e-5/self.dt)
            
            ### Widen time of peak by mu_bar scaling ###
            self.time_factor = int(np.ceil((self.mu_bar**-0.75)/self.dt))
            
            self.peak_start_idx = np.array([(math.floor(i/10)*10)-self.time_factor for \
                                            i in self.peak_start_idx])
            self.peak_end_idx = np.array([(math.ceil(i/10)*10)+self.time_factor for \
                                            i in self.peak_end_idx])
            
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
        If the location of max curvature is away from the center, then the data is reco
        """
        if self.elastic_peak_idx.size > 0:
            self.curvature_threshold = self.mu_bar**0.15
            # self.curvature_threshold = 8 #Minimum value or curvature to be considered for
            
            self.curvature_vals_list = []
            for idx in self.adj_elastic_peak_idx:
                self.curv_peak_idx,_ = find_peaks(self.k_curvature[:,idx],height = self.curvature_threshold,distance = 10) #When elastic energy peaks, find peak curvature
                self.curv_peak_widths_results = peak_widths(self.k_curvature[:,idx],self.curv_peak_idx,rel_height = 0.99) #Find peak start and end based on 99% of height value
                theoretical_radius = self.curv_peak_widths_results[0]*self.ds/np.pi
                #Check if curvature peaks in 1 location along filament and not near center
                if self.curv_peak_idx.shape[0] == 1:
                    if self.curv_peak_idx[0]  >= 0.35*self.N and self.curv_peak_idx[0] <= 0.65*self.N:
                        # logging.warning(
                        #     "The location of max curvature is too close to the center and not a J-shape! s index of max curvature: {}. Results will be appended as NaN".format(
                        #         curv_peak_idx[0]))
                        curvature_results = {"Elastic Energy Time Index": [idx],
                                             "Curvature Peak Filament Index": int(self.curv_peak_idx),
                                             "Max Curvature Value": np.nan,
                                             "Radius of Bending-2":np.nan}
                    elif self.curv_peak_idx[0]  <= 0.07*self.N or self.curv_peak_idx[0] >= 0.93*self.N:
                        # logging.warning(
                        #     "The location of max curvature is too close to the end and not a J-shape! s index of max curvature: {}. Results will be appended as NaN".format(
                        #         curv_peak_idx[0]))
                        curvature_results = {"Elastic Energy Time Index": [idx],
                                             "Curvature Peak Filament Index": int(self.curv_peak_idx),
                                             "Max Curvature Value": np.nan,
                                             "Radius of Bending-2":np.nan}
                    else:
                        max_curv_val = self.k_curvature[self.curv_peak_idx[0],idx]
                        curvature_results = {"Elastic Energy Time Index": [idx],
                                             "Curvature Peak Filament Index": int(self.curv_peak_idx),
                                             "Max Curvature Value": max_curv_val,
                                             "Radius of Bending-2":theoretical_radius}
                else:
                    # logging.warning("Error! There are two peaks at t = {}. Curvature data will be written as NaN.".format(self.time_data[idx]))
                    # self.plot_f_pos_curvature_debug(curv_peak_idx,idx)
                    curvature_results = {"Elastic Energy Time Index": [idx],
                                         "Curvature Peak Filament Index": np.nan,
                                         "Max Curvature Value": np.nan,
                                         "Radius of Bending-2":np.nan}
                self.curvature_vals_list.append(curvature_results)
            if len(self.curvature_vals_list) > 0:
                self.curvature_vals_df = pd.concat(
                    [pd.DataFrame.from_dict(i) for i in self.curvature_vals_list],ignore_index = True)
                self.curvature_vals_df.set_index("Elastic Energy Time Index",inplace = True)
                self.curvature_vals_df.dropna(how = 'any',inplace = True)
            else:
                self.curvature_vals_df = pd.DataFrame()
        logging.info("Finished identifying the location of maximum curvature along the filament.")
                
        
    def plot_fpos_ee_debug(self,time_low,time_high,i,j):
        """
        This function plots the filament positions and elastic energy data
        where there's conflict between the peak start/end values.
        
        Inputs:
        
        time_low:           Time end index of previous elastic peak. 
        time_high:          Time start index of current elastic peak. 
        i:                  Index of previous elastic peak.
        j:                  Index of current elastic peak. 
        """
        fig,axes = plt.subplots(ncols =2, figsize = (18,10))
        
        axes[0].plot(self.position_data[:,0,i],self.position_data[:,1,i],'black',
                     linewidth = 1.2,label = r'$t^{{Br}} = {0:.3e}$'.format(self.time_data[i]))
        axes[0].plot(self.position_data[:,0,j],self.position_data[:,1,j],'green',linewidth = 1.2,
                     label = r'$t^{{Br}} = {0:.3e}$'.format(self.time_data[j]))
        
        
        axes[1].plot(self.time_data[time_low:time_high],
                  self.elastic_data[time_low:time_high],'black',linewidth = 1.2)
        axes[1].plot(self.time_data[i],self.elastic_data[i],
                  color = 'red',marker= 'o',markersize = 7)
        axes[1].plot(self.time_data[j],self.elastic_data[j],
                  color = 'red',marker= 'o',markersize = 7)
        
        if self.channel_height == 0.25:
            axes[0].set_ylim(-0.3,0.3)
        elif self.channel_height == 0.5:
            axes[0].set_ylim(-0.6,0.6)
        elif self.channel_height == 0.75:
            axes[0].set_ylim(-0.8,0.8)
        axes[0].set_xlim(-0.6,0.6)
        axes[0].axhline(y = self.channel_height,color = 'gray',linewidth = 1.5)
        axes[0].axhline(y = -self.channel_height,color = 'gray',linewidth = 1.5)
        axes[0].tick_params(axis='x', which='major', labelsize = 16, size = 5,width = 4)
        axes[0].tick_params(axis='y', which='major',labelsize = 16, size = 5,width = 4)
        axes[0].set_xlabel('x',fontsize = 20,labelpad = 10)
        axes[0].set_ylabel('y',fontsize = 20,labelpad = 10)
        axes[0].set_title("Position",fontsize = 20,pad = 10)
        axes[0].legend()
        
        axes[1].set_xlabel(r'$t^{Br}$',fontsize=25,labelpad = 20)
        axes[1].set_ylabel(r'$E_{elastic}$',fontsize=25,labelpad = 20)
        axes[1].xaxis.offsetText.set_fontsize(18)
        axes[1].set_xlim(self.time_data[time_low]*0.99,self.time_data[time_high]*1.01)
        axes[1].ticklabel_format(axis="x", style="sci", scilimits=(-5,-5))
        axes[1].tick_params(axis='x', which='major', labelsize = 20, size = 6,width = 5)
        axes[1].tick_params(axis='y', which='major',labelsize = 20, size = 6,width = 5)
        axes[1].set_title(r"$E_{{elastic}}$ Landscape")
        if self.flow_type == 'Poiseuille':
            flow_profile = r"$U_{{x}} = {0:.2f}\left(1-y^{{2}}/{1}^{{2}}\right)$".format(self.U_centerline,self.channel_height)
        elif self.flow_type == 'Shear':
            flow_profile = r"$U_{{x}} = y$"
        elif self.flow_type == 'Kolmogorov':
            flow_profile = r"$U_{{x}} =$ sin$\left({0:.0f}\pi y\right)$".format(self.kflow_freq)
        fig.suptitle(r"{0}" "\n" r"$\bar{{\mu}} = {1:.0e} |$ Rep = {2} | $\Delta y_{{0}} = {3}$".format(
            flow_profile,self.mu_bar,self.rep_number,self.vert_displ),fontsize = 20,y = 0.990)
        plt.show()

              
    def plot_f_pos_curvature_debug(self,idx):
        """
        This function plots the locations of high curvature along the filament if
        there exists more than 1 point of high curvature.
        
        Inputs:
        
        idx:                Index of self.elastic_peak_idx of interest.
        """
        fig,axes = plt.subplots(ncols=2,figsize=(18,10))
        axes[0].plot(self.position_data[:,0,idx],self.position_data[:,1,idx],'black',linewidth = 1.5)
        axes[1].plot(self.s,self.k_curvature[:,idx],'black',linewidth = 1.5)
        
        ### Give markers as to where the locations of high curvature are ###
        for high_curv_loc in self.curv_peak_idx:
            axes[0].plot(self.position_data[high_curv_loc,0,idx],self.position_data[high_curv_loc,1,idx],
                          color = 'red',marker = 'd',markersize = 10)
            axes[1].plot(self.s[high_curv_loc],self.k_curvature[high_curv_loc,idx],color = 'red',marker = 'd',markersize = 10)
        if self.channel_height == 0.25:
            axes[0].set_ylim(-0.3,0.3)
        elif self.channel_height == 0.5:
            axes[0].set_ylim(-0.6,0.6)
        elif self.channel_height == 0.75:
            axes[0].set_ylim(-0.8,0.8)
        axes[0].set_xlim(-0.6,0.6)
        axes[0].axhline(y = self.channel_height,color = 'gray',linewidth = 1.5)
        axes[0].axhline(y = -self.channel_height,color = 'gray',linewidth = 1.5)
        axes[0].tick_params(axis='x', which='major', labelsize = 16, size = 5,width = 4)
        axes[0].tick_params(axis='y', which='major',labelsize = 16, size = 5,width = 4)
        axes[0].set_xlabel('x',fontsize = 20,labelpad = 10)
        axes[0].set_ylabel('y',fontsize = 20,labelpad = 10)
        axes[0].set_title("Position",fontsize = 20,pad = 10)
        axes[0].set_aspect((axes[0].get_xlim()[1] - axes[0].get_xlim()[0])/(axes[0].get_ylim()[1] - axes[0].get_ylim()[0]))
        
        ### Plot ends of filament ###
        end_points = [0,-1]
        end_point_txt = [r"$s = -\frac{1}{2}$",r"$s = \frac{1}{2}$"]
        for ep_idx,ep in enumerate(end_points):
            axes[0].plot(self.position_data[ep,0,idx],self.position_data[ep,1,idx],
                         color = 'cyan',marker = 'o',markersize = 7)
            axes[0].annotate('{}'.format(end_point_txt[ep_idx]),
                              xy = (self.position_data[ep,0,idx],
                                    self.position_data[ep,1,idx]),
                              xytext = (-30,-60),textcoords = 'offset points',color = 'black',fontsize = 15,
                              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=1,linewidth = 1.5),
                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.8', 
                                    color='magenta',linewidth = 2))               
        axes[1].set_xlim(-0.6,0.6)
        axes[1].axhline(y = self.curvature_threshold,color = 'gray',linewidth = 1.5)
        axes[1].tick_params(axis='x', which='major', labelsize = 16, size = 5,width = 4)
        axes[1].tick_params(axis='y', which='major',labelsize = 16, size = 5,width = 4)
        axes[1].set_xlabel(r'$s$',fontsize = 20,labelpad = 10)
        axes[1].set_ylabel(r'$k(s)$',fontsize = 20,labelpad = 10)
        axes[1].set_title("Curvature",fontsize = 20,pad = 10)
        axes[1].set_aspect((axes[1].get_xlim()[1] - axes[1].get_xlim()[0])/(axes[1].get_ylim()[1] - axes[1].get_ylim()[0]))
        if self.flow_type == 'Poiseuille':
            flow_profile = r"$U_{{x}} = {0:.2f}\left(1-y^{{2}}/{1}^{{2}}\right)$".format(self.U_centerline,self.channel_height)
        elif self.flow_type == 'Shear':
            flow_profile = r"$U_{{x}} = y$"
        elif self.flow_type == 'Kolmogorov':
            flow_profile = r"$U_{{x}} =$ sin$\left({0:.0f}\pi y\right)$".format(self.kflow_freq)
        fig.suptitle(r"$t^{{Br}} = {0:.3e} $| {1}" "\n" r"$\bar{{\mu}} = {2:.0e} |$ Rep = {3}".format(
            self.time_data[idx],flow_profile,self.mu_bar,self.rep_number),fontsize = 20,y = 0.990)
        plt.show()
        

class all_ensemble_data():
    """
    This class will store all of the ensemble data temporarily in a dictionary
    before converting it into a DataFrame.
    """
    
    def __init__(self,output_dir):
        """
        This initialization function will just create an empty list for all ensemble
        data to be stored in.
        """
        self.all_ensemble_loc_data_list = []
        self.all_ensemble_curvature_data_list = []
        self.output_dir = output_dir
    def add_curvature_data(self,ensemble_class):
        """
        This function will convert each individual ensemble data (present in a DataFrame)
        into a dictionary before appending it to the list. This function specifically pertains
        to net displacement and curvature data.
        
        Inputs:
        
        ensemble_class:         Class variable that has all of the information pertaining
                                to a particular ensemble.
        """
        if ensemble_class.elastic_peak_idx.size > 0:
            for i,ee_peak_idx in enumerate(ensemble_class.curvature_vals_df.index.values):
                
                true_idx = np.where(ensemble_class.adj_elastic_peak_idx == ee_peak_idx)[0][0]
                start_com_y = ensemble_class.center_mass[ensemble_class.adj_peak_start_idx[true_idx],1]
                end_com_y = ensemble_class.center_mass[ensemble_class.adj_peak_end_idx[true_idx],1]
                start_tc_y = ensemble_class.position_data[ensemble_class.true_center_loc,1,ensemble_class.adj_peak_start_idx[true_idx]]
                end_tc_y = ensemble_class.position_data[ensemble_class.true_center_loc,1,ensemble_class.adj_peak_end_idx[true_idx]]
                peak_curv_loc = int(ensemble_class.curvature_vals_df.loc[ee_peak_idx,'Curvature Peak Filament Index'])
                max_curv_val = ensemble_class.curvature_vals_df.loc[ee_peak_idx,'Max Curvature Value']
                radius_bending_2 = ensemble_class.curvature_vals_df.loc[ee_peak_idx,'Radius of Bending-2']
                
                net_com_y = np.abs(end_com_y) - np.abs(start_com_y)
                net_tc_y = np.abs(end_tc_y)  - np.abs(start_tc_y)
                
                
                if np.isnan(peak_curv_loc):
                    s_peak_curv_loc = np.array([np.nan])
                else:
                    s_peak_curv_loc = np.array([ensemble_class.s[peak_curv_loc]])
                
                ### Find radius values
                if np.isnan(max_curv_val):
                    radius_bending_1 = np.nan
                    radius_bending_2 = np.nan
                else:
                    radius_bending_1 = max_curv_val**-1
                    
                ### Find time points between elastic peaks ###
                if i != 0:
                    self.curr_ee_peak_time = ensemble_class.time_data[ensemble_class.adj_elastic_peak_idx[true_idx]]
                    self.prev_ee_peak_time = ensemble_class.time_data[ensemble_class.adj_elastic_peak_idx[i-1]]
                    ee_peak_time_diff = self.curr_ee_peak_time - self.prev_ee_peak_time
                else:
                    ee_peak_time_diff = np.array([0])
                self.indiv_ensemble_curv_dict = {"Rigidity Suffix":ensemble_class.rigidity_profile,
                                            "Mu_bar": ensemble_class.mu_bar,
                                            "N": ensemble_class.N,
                                            "Rep Number": ensemble_class.rep_number,
                                            'Channel Height': ensemble_class.channel_height,
                                            'Poiseuille U Centerline': ensemble_class.U_centerline,
                                            'Kolmogorov Frequency': ensemble_class.kflow_freq,
                                            'Kolmogorov Phase Text': ensemble_class.kflow_phase_text,
                                            'Kolmogorov Phase Value': ensemble_class.kflow_phase_val,
                                            'Starting Vertical Displacement': ensemble_class.vert_displ,
                                            'Steric Velocity Exponential Coefficient':ensemble_class.velo_exp,
                                            'Steric Velocity Gap Criteria': ensemble_class.velo_gap,
                                            'Flow Type': ensemble_class.flow_type,
                                            'Brownian Use':ensemble_class.brownian_use,
                                            'Sterics Use': ensemble_class.sterics_use,
                                            "Max Elastic Energy Index Value": ee_peak_idx,
                                            "Time of Max Elastic Energy": ensemble_class.time_data[ee_peak_idx],
                                            "Elastic Energy Value at Peak": ensemble_class.elastic_data[ee_peak_idx],
                                            "Start Time of Elastic Energy Peak": ensemble_class.time_data[ensemble_class.adj_peak_start_idx[true_idx]],
                                            "End Time of Elastic Energy Peak": ensemble_class.time_data[ensemble_class.adj_peak_end_idx[true_idx]],
                                            "Start Time Index of Elastic Energy Peak": ensemble_class.adj_peak_start_idx[true_idx],
                                            "End Time Index of Elastic Energy Peak": ensemble_class.adj_peak_end_idx[true_idx],
                                            "Time Difference Between Current and Previous Rotation":ee_peak_time_diff,
                                            "Starting COM-y":start_com_y,
                                            "COM-y at Elastic Peak": ensemble_class.center_mass[ee_peak_idx,1],
                                            "Ending COM-y": end_com_y,
                                            "Starting True Center-y":start_tc_y,
                                            "True Center-y at Elastic Peak":ensemble_class.position_data[ensemble_class.true_center_loc,1,ee_peak_idx],
                                            "Ending True Center-y":end_tc_y,
                                            "Net COM-y": net_com_y,
                                            "Net True Center-y": net_tc_y,
                                            "Rotation Duration": ensemble_class.adj_elastic_peak_duration[true_idx],
                                            "Drift Rotation Velocity": net_com_y/ensemble_class.adj_elastic_peak_duration[true_idx],
                                            "Peak Curvature Index Value": peak_curv_loc,
                                            "Location of Peak Curvature": s_peak_curv_loc,
                                            "Peak Curvature Value":max_curv_val,
                                            "Radius of Bending-1": radius_bending_1,
                                            "Radius of Bending-2": radius_bending_2}
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
                self.all_ensemble_k_df.loc[group_df.index.values,'Local Shear Rate'] = group_df['Kolmogorov Frequency'] * np.pi * np.cos(group_df['Kolmogorov Frequency']*np.pi * group_df['Starting COM-y'])
                
                
        self.all_ensemble_k_df['Displacement Type'] = self.all_ensemble_k_df['Net COM-y'].apply(lambda x: self.classify_displ_type(x))
        self.all_ensemble_k_df['ABS Net COM-y'] = np.abs(self.all_ensemble_k_df['Net COM-y'])
        
        self.all_ensemble_k_df['Depletion Layer Thickness'] = 1.10*(self.all_ensemble_k_df['Mu_bar']**-0.125)
        
        ### Filter Data for subsequent analysis of net displacement of free J-shapes ###
        self.all_ensemble_k_df = self.all_ensemble_k_df[(self.all_ensemble_k_df['Starting COM-y'] >= 0.05) &\
                                                (self.all_ensemble_k_df['Mu_bar'] > 25000) &\
                                                    (self.all_ensemble_k_df['Starting COM-y'] <\
                                                     (self.all_ensemble_k_df['Channel Height'] - self.all_ensemble_k_df['Depletion Layer Thickness']))]

        ### Give String label for Mu_bar values ###
        self.all_ensemble_k_df['Mu_bar String'] = self.all_ensemble_k_df['Mu_bar'].apply(lambda x: self.mu_bar_label(x))
        
        ### Calculated Position-dependent Mu_bar ###
        self.all_ensemble_k_df.loc[:,'Mu_bar times Starting COM-y'] =  self.all_ensemble_k_df.loc[:,'Mu_bar'] *  self.all_ensemble_k_df.loc[:,'Starting COM-y']
        
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
    
    def mu_bar_label(self,mu_bar):
        """
        This function gives a string label for the mu_bar value.
        
        Inputs:
            
        mu_bar:             Mu_bar value that will be assigned a string value.
        """
        
        if mu_bar == 50000:
            new_mu_bar = r'$5 \times 10^{4}$'
        elif mu_bar == 100000:
            new_mu_bar = r'$1 \times 10^{5}$'
        elif mu_bar == 200000:
            new_mu_bar = r'$2 \times 10^{5}$'
        elif mu_bar == 500000:
            new_mu_bar = r'$5 \times 10^{5}$'
        return new_mu_bar
        
    def save_data(self,file_name):
        """
        This method saves the CSV file that contains all information for the 
        filament flip events.
        """
        self.all_ensemble_k_df.to_csv(
            os.path.join(self.output_directory,'{}.csv'.format(file_name)))
    
    def create_dir(self,dir_):
        """
        This method creates a new directory if it doesn't already exist. Note
        that every instance of this method will import the create_dir module
        from the project directory folder.
        """
        from misc.create_dir import create_dir
        create_dir(dir_)
        
    def plot_flip_net_displacement_violin(self,df_name,plot_name,flow_type):
        """
        This method will perform a 2-way ANOVA on the filament net displacement
        of up & down flip events based on the various mu_bars used for the 
        simulations. After calculating the 2-way ANOVA statistics 
        (with appropriate ad-hoc) correction tests, it will generate violin 
        plots that show the distribution of the net center of mass 
        displacement values. Note that every instance of this method will import the 
        Plot_Net_Displacement_Mu_bar module from the project directory.
        """
        from calculations.ANOVA2_netycom_flipdir import ANOVA2_netycom_flipdir
        from plotting.net_ycom_mu_bar_violin import net_ycom_mu_bar_violin
        violin_dir = os.path.join(self.output_dir,'violin_plots_net_displ')
        self.create_dir(violin_dir)
        
        self.net_displ_stats = ANOVA2_netycom_flipdir(
            input_file = self.all_ensemble_k_df,
            output_directory = violin_dir,
            file_name = df_name)
        net_ycom_mu_bar_violin(
            input_file = self.all_ensemble_k_df,
            p_vals_df = self.net_displ_stats,
            flow_type = flow_type,
            output_directory = violin_dir,
            file_name = plot_name)
        
    def plot_flip_radius_violin(self,df_name,plot_name,flow_type):
        """
        This method will perform a 2-way ANOVA on the filament radius
        of up & down flip events based on the various mu_bars used for the 
        simulations. After calculating the 2-way ANOVA statistics 
        (with appropriate ad-hoc) correction tests, it will generate violin 
        plots that show the distribution of the net center of mass 
        displacement values. Note that every instance of this method will import the 
        Plot_Net_Displacement_Mu_bar module from the project directory.
        """
        from calculations.ANOVA2_radius_flipdir import ANOVA2_radius_flipdir
        from plotting.radius_mu_bar_violin import radius_mu_bar_violin
        violin_dir = os.path.join(self.output_dir,'violin_plots_radius')
        self.create_dir(violin_dir)
        
        self.radius_stats = ANOVA2_radius_flipdir(
            input_file = self.all_ensemble_k_df,
            output_directory = violin_dir,
            file_name = df_name)
        radius_mu_bar_violin(
            input_file = self.all_ensemble_k_df,
            p_vals_df = self.radius_stats,
            flow_type = flow_type,
            output_directory = violin_dir,
            file_name = plot_name)
        
    def radius_mu_bar_scaling(self,flow_type,file_name):
        """
        With the Dataframe that contains all of the filament flip information,
        it will plot the radius at maximum curvature as a function of mu_bar
        as a scatterplot. Note that every instance of this method will import
        the Plot_Radius_Mu_bar module from the project directory.
        """
        from plotting.Plot_Radius_Mu_bar import plot_radius_mu_bar
        
        radius_mu_bar_reg_dir = os.path.join(self.output_dir,'rad_mu_bar_regular')
        self.create_dir(radius_mu_bar_reg_dir)
        
        plot_radius_mu_bar(
            input_file = self.all_ensemble_k_df,
            output_directory = radius_mu_bar_reg_dir,
            flow_type = flow_type,
            file_name = file_name)
        
    def radius_adjusted_mu_bar_scaling(self,file_name):
        """
        With the Dataframe that contains all of the filament flip information,
        it will plot the radius at maximum curvature as a function of mu_bar
        as a scatterplot. Note that every instance of this method will import
        the Plot_Radius_Adjusted_Mu_bar module from the project directory.
        """
        from plotting.Plot_Radius_Adjusted_Mu_bar import plot_radius_adj_mu_bar
        radius_mu_bar_adj_dir = os.path.join(self.output_dir,'rad_mu_bar_adj')
        self.create_dir(radius_mu_bar_adj_dir)
        
        plot_radius_adj_mu_bar(
            input_file = self.all_ensemble_k_df,
            output_directory = radius_mu_bar_adj_dir,
            file_name = file_name)
        
    def radius_mu_bar_scaling_subplots(self,file_name):
        """
        This method generates 2 subplots that show the relationship between the 
        calculated curvature as a function of mu_bar times filament center of
        mass prior to the flip event. It also generates 2 inset plots that show
        the curvature as a function of just mu_bar.
        """
        from plotting.radius_mu_bar_scaling_subplots import radius_mu_bar_scaling_subplots
        radius_mu_bar_sp_dir = os.path.join(self.output_dir,'rad_mu_bar_all')
        self.create_dir(radius_mu_bar_sp_dir)
        
        def radius_scaling(x,a): return a*x**(-0.25)
        
        input_df = self.all_ensemble_k_df.copy()
        fil_poi_df = input_df[input_df['Flow Type'] == 'Poiseuille']
        fil_shear_df = input_df[input_df['Flow Type'] == 'Shear']
        
        poi_intercept_1 = curve_fit(radius_scaling,
                                    xdata = fil_poi_df['Mu_bar times Starting COM-y'].to_numpy(),
                                    ydata = fil_poi_df['Radius of Bending-1'].to_numpy())[0]
        poi_intercept_2 = curve_fit(radius_scaling,
                                    xdata = fil_poi_df['Mu_bar'].to_numpy(),
                                    ydata = fil_poi_df['Radius of Bending-1'].to_numpy())[0]
        
        shear_intercept_1 = curve_fit(radius_scaling,
                                    xdata = fil_shear_df['Mu_bar times Starting COM-y'].to_numpy(),
                                    ydata = fil_shear_df['Radius of Bending-1'].to_numpy())[0]
        shear_intercept_2 = curve_fit(radius_scaling,
                                    xdata = fil_shear_df['Mu_bar'].to_numpy(),
                                    ydata = fil_shear_df['Radius of Bending-1'].to_numpy())[0]
        slope_intercept_fit_params = {"Poiseuille Spatial Intercept": poi_intercept_1,
                      "Poiseuille Regular Intercept": poi_intercept_2,
                      "Shear Spatial Intercept": shear_intercept_1,
                      "Shear Regular Intercept": shear_intercept_2,
                      "Poiseuille Spatial Slope": -0.25,
                      "Poiseuille Regular Slope": -0.25,
                      "Shear Spatial Slope": -0.25,
                      "Shear Regular Slope": -0.25}
        radius_mu_bar_scaling_subplots(
            input_file = self.all_ensemble_k_df,
            fit_params = slope_intercept_fit_params,
            output_directory = radius_mu_bar_sp_dir,
            file_name = file_name)
#%% Main Body of script

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_directory",
                        help="Specify the path to the directory that contains this script and all other relevant scripts",
                    type = str)
    parser.add_argument("output_directory",
                        help="Specify the parent directory of the Migration Data",
                    type = str)
    parser.add_argument("flow_type",
                        help = "Specify what kind of background flow(s) that input data are",
                        type = str)
    parser.add_argument("--input_directory",'-input_dir',nargs = '+',
                        help="Specify the parent directory of the Migration Data",
                    type = str)
    parser.add_argument("--statistical_assessment",'-stats',
                        help = "Specify if you want to perform statistical analyses of the radii of bending (requires statsannotations library)",
                        action = 'store_true',default = False)
    
    # args = parser.parse_args()
    
    args = parser.parse_args(['C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Scripts//03_Flip_Data//',
                              'C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Scripts//03_Flip_Data//02_Actual_Results//Poiseuille_Shear_Walls//',
                              "Poiseuille_Shear",
                              '--input_directory','C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Remote_Data//Poiseuille_Flow_Walls//',
                              'C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Remote_Data//Shear_Flow_Walls//'])
    
    ### Just Poiseuille flow with statistical analysis
    # args = parser.parse_args(['C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Scripts//03_Flip_Data//',
    #                           'C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Scripts//03_Flip_Data//02_Actual_Results//Poiseuille_Flow_Walls//',
    #                           "Poiseuille",
    #                           '--input_directory','C://Users//super//OneDrive - University of California, Davis//Research//00_Projects//02_Shear_Migration//00_Remote_Data//Poiseuille_Flow_Walls//',
    #                           '-stats'])
    
    os.chdir(args.project_directory)
    #Import other functions
    from calculations.first_derivative import first_derivative
    from calculations.second_derivative import second_derivative
    from calculations.center_of_mass import center_of_mass
    # from calculations.adjust_position_data import adjust_position_data

    
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s|%(filename)s|%(levelname)s|%(message)s',
            datefmt="%A, %B %d at %I:%M:%S %p")
    ### Read in Data ###
    all_ensbl_dat = all_ensemble_data(args.output_directory)
    if args.input_directory:
        for dir_ in args.input_directory:
            for root,dirs,files in os.walk(dir_):
                for subdir_ in dirs:
                    check_file = os.path.join(root,subdir_,'filament_allstate.npy')
                    check_file2 = os.path.join(root,subdir_,'parameter_values.csv')
                    if os.path.exists(check_file) and os.path.exists(check_file2):
                        load_check_file2 = pd.read_csv(check_file2,index_col = 0,header = 0)
                        if "R" in subdir_ and float(load_check_file2.loc['Channel Upper Height','Value']) == 0.5 and\
                            (float(load_check_file2.loc['Vertical Displacement','Value']) > 0):
                            match = re.search(r"R_(\d{1,})",subdir_)
                            if match and int(match.group(1)) <= 10:
                                
                                path_to_dir = os.path.join(root,subdir_)
                                replicate_number = int(match.group(1))
                                
                                ensmbl_dat = indiv_ensemble_data(path_to_dir,replicate_number)
                                ensmbl_dat.find_ensemble_parameters()
                                logging.info(
                                    "Now finished reading in ensemble data for Mu_bar = {} | Replicate ={} | U_centerline = {}".format(
                                        ensmbl_dat.mu_bar,ensmbl_dat.rep_number,ensmbl_dat.U_centerline))
                                ### Curvature Data ###
                                ensmbl_dat.calculate_curvature()
                                ensmbl_dat.detect_elastic_peaks()
                                ensmbl_dat.find_max_curvature()
                                all_ensbl_dat.add_curvature_data(ensmbl_dat)
                            
     
    ### Process Curvature Data ###
    all_ensbl_dat.compile_position_curvature_data()
     
    ### Plot Commands ###
    if args.statistical_assessment:
        # all_ensbl_dat.plot_flip_net_displacement_violin(df_name = 'poiseuille_walls_net_displ_stat',
        #                                                 plot_name = 'poiseuille_violin_plot',
        #                                                 flow_type = 'Poiseuille')
        all_ensbl_dat.plot_flip_radius_violin(df_name = 'poiseuille_walls_radius_stat',
                                                        plot_name = 'poiseuille_violin_plot',
                                                        flow_type = 'Poiseuille')
    
    
    # all_ensbl_dat.radius_mu_bar_scaling(flow_type = 'Poiseuille',
    #                                               file_name = 'Poiseuille_walls_H_0p50')
    # all_ensbl_dat.radius_adjusted_mu_bar_scaling(file_name = 'Poiseuille_walls_H_0p50')
    
    if args.flow_type == 'Poiseuille_Shear':
        all_ensbl_dat.radius_mu_bar_scaling_subplots(file_name = 'Poiseuille_Shear_walls_H_0p50')
    logging.info("Code has finished compiling all curvature data.")


