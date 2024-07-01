# -*- coding: utf-8 -*-
"""
FILE NAME:      Plot_All_LP.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        create_dir.py; first_derivative.py; second_derivative.py; 
                center_of_mass.py

DESCRIPTION:    For a given upward and downward flip motion in Poiseuille or 
                shear flow, this script will plot snapshots of a particular
                attribute of the filament during the motion; these plots are 
                arranged in an array-like format. 

INPUT
FILES(S):       1) .NPY file that contains the filament position at every 
                timepoint.
                2) .NPY file that contains the timepoint of each attribute
                of the filament.
                3) .NPY file that contains the total filament elastic energy
                at every timepoint.
                4) .NPY file that contains the filament extra particle stress
                tensor values at every timepoint.
                5) .NPY file that contains the filament tension profile at
                every timepoint. 
                6. .CSV file that contains all of the parameters used for the
                filament simulations.

OUTPUT
FILES(S):       1) .PNG file(s) of the particular attribute of the filament.


INPUT
ARGUMENT(S):    N/A; this function is not formatted to run from the command line.
                However, you do need to specify the following:
                    
                1) Project directory: The path to the main directory that has 
                the script to create a new directory.
                2) Upward flip directory: The path to the upward flip directory 
                that has contains the .NPY data files needed to plot.
                3) Downward flip directory: The path to the upward flip 
                directory that has contains the .NPY data files needed to plot.
                4) Output Directory: The path to the directory that will contain
                the resulting .PNG files. If it currently doesn't exist, it will
                be created.


CREATED:        21Apr23

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
import re, sys, os, logging, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import argrelmin, argrelmax
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})




#%%

class up_down_data():
    def __init__(self,up_dir,down_dir,output_dir):
        self.up_dir = up_dir
        self.down_dir = down_dir
        self.output_dir = output_dir
        create_dir(self.output_dir)
        
        ##### Up Data #####
        self.up_position_data = np.load(os.path.join(self.up_dir,'filament_allstate.npy'))
        self.up_tension_data = np.load(os.path.join(self.up_dir,'filament_tension.npy'))
        self.up_elastic_data = np.load(os.path.join(self.up_dir,'filament_elastic_energy.npy'))
        self.up_stress_data = np.load(os.path.join(self.up_dir,'filament_stress_all.npy'))
        self.up_time_data = np.load(os.path.join(self.up_dir,'filament_time_vals_all.npy'))
        self.up_param_df = pd.read_csv(os.path.join(self.up_dir,'parameter_values.csv'),
                                    index_col = 0,header = 0)
        
        ### Relevant Parameters ###
        self.up_mu_bar = float(self.up_param_df.loc['Mu_bar','Value'])
        self.up_s = np.linspace(float(self.up_param_df.loc['Filament s start','Value']),
                                float(self.up_param_df.loc['Filament s end','Value']),
                                int(self.up_param_df.loc['N','Value']))
        self.up_N = int(self.up_param_df.loc['N','Value'])
        self.up_ds = 1/(self.up_N - 1)
        self.up_vert_displ = float(self.up_param_df.loc['Vertical Displacement','Value'])
        self.up_t_end = float(self.up_param_df.loc['Simulation End Time','Value'])
        self.up_dt = float(self.up_param_df.loc['Array Time Step Size','Value'])
        self.up_tc = np.where(self.up_s == 0)[0][0]
        
            
        ### Process Up Data ###
        #Center of Mass: rows are time points, columns are position coordinates
        self.up_com_data = center_of_mass(position_data = self.up_position_data, 
                                          position = 0, dim = 3, adj_centering = True,
                                          adj_translation = True, transl_val = self.up_vert_displ)
        
        #True center: rows are time points, columns are position coordinates
        self.up_tc_data = self.up_position_data[self.up_tc,:,:].T 
        self.up_time_data_adj = self.up_time_data/self.up_time_data.max()
        self.up_N1_data = self.up_stress_data[1,1,:] - self.up_stress_data[0,0,:]
        self.up_N2_data = self.up_stress_data[2,2,:] - self.up_stress_data[1,1,:]
        if (self.up_stress_data[1,0,:] == self.up_stress_data[0,1,:]).all():
            self.up_Sxy_data = self.up_stress_data[1,0,:]
        else:
            logging.warning("Error! The off-diagonal components of the stress tensor are not equal to each other. Please check your stress tensor!")
            sys.exit(1)
            
        # Calculate derivatives of position & com velocity #
        self.up_dxds_data = first_derivative(base_array = self.up_position_data,
                                        deriv_size = self.up_ds,axis = 0,
                                        ar_size = self.up_position_data.shape[0], 
                                        dim = 3)
        
        
        self.up_d2xds2_data = second_derivative(base_array = self.up_position_data,
                                        deriv_size = self.up_ds,axis = 0,
                                        ar_size = self.up_position_data.shape[0], 
                                        dim = 3)
        
        self.up_ducomdt_data = first_derivative(base_array = self.up_com_data,
                                        deriv_size = self.up_dt,axis = 0,
                                        ar_size = self.up_com_data.shape[0], 
                                        dim = 2)
        
        self.d2ucomdt2_up_data = second_derivative(base_array = self.up_com_data,
                                        deriv_size = self.up_dt,axis = 0,
                                        ar_size = self.up_com_data.shape[0], 
                                        dim = 2)
        
        # Calculate Maximum curvature along filament #
        self.up_curvature_data = (np.abs((self.up_dxds_data[:,0,:]*self.up_d2xds2_data[:,1,:]) -\
                                         (self.up_dxds_data[:,1,:]*self.up_d2xds2_data[:,0,:])))/\
            (((self.up_dxds_data[:,0,:]**2)+(self.up_dxds_data[:,1,:]**2))**(1.5))
            
        self.up_max_curv_data = self.up_curvature_data.max(axis = 0)
        
        #Calculate angle of orientation of filament
        self.up_angle_orientation = np.rad2deg(np.mod(np.arctan2(self.up_dxds_data[:,1,:],self.up_dxds_data[:,0,:]),2*np.pi))
        
        ##### Down Data #####
        self.down_position_data = np.load(os.path.join(self.down_dir,'filament_allstate.npy'))
        self.down_tension_data = np.load(os.path.join(self.down_dir,'filament_tension.npy'))
        self.down_elastic_data = np.load(os.path.join(self.down_dir,'filament_elastic_energy.npy'))
        self.down_stress_data = np.load(os.path.join(self.down_dir,'filament_stress_all.npy'))
        self.down_time_data = np.load(os.path.join(self.down_dir,'filament_time_vals_all.npy'))
        self.down_param_df = pd.read_csv(os.path.join(self.down_dir,'parameter_values.csv'),
                                    index_col = 0,header = 0)
        
        ### Relevant Parameters ###
        self.down_mu_bar = float(self.down_param_df.loc['Mu_bar','Value'])
        self.down_s = np.linspace(float(self.down_param_df.loc['Filament s start','Value']),
                                float(self.down_param_df.loc['Filament s end','Value']),
                                int(self.down_param_df.loc['N','Value']))
        self.down_N = int(self.down_param_df.loc['N','Value'])
        self.down_ds = 1/(self.down_N - 1)
        self.down_vert_displ = float(self.down_param_df.loc['Vertical Displacement','Value'])
        self.down_t_end = float(self.down_param_df.loc['Simulation End Time','Value'])
        self.down_dt = float(self.down_param_df.loc['Array Time Step Size','Value'])
        self.down_tc = np.where(self.down_s == 0)[0][0]
        
            
        ### Process down Data ###
        #Center of Mass: rows are time points, columns are position coordinates
        self.down_com_data = center_of_mass(position_data = self.down_position_data, 
                                          position = 0, dim = 3, adj_centering = True,
                                          adj_translation = True, transl_val = self.down_vert_displ)
        
        #True center: rows are time points, columns are position coordinates
        self.down_tc_data = self.down_position_data[self.down_tc,:,:].T
        self.down_time_data_adj = self.down_time_data/self.down_time_data.max()
        self.down_N1_data = self.down_stress_data[1,1,:] - self.down_stress_data[0,0,:]
        self.down_N2_data = self.down_stress_data[2,2,:] - self.down_stress_data[1,1,:]
        if (self.down_stress_data[1,0,:] == self.down_stress_data[0,1,:]).all():
            self.down_Sxy_data = self.down_stress_data[1,0,:]
        else:
            logging.warning("Error! The off-diagonal components of the stress tensor are not equal to each other. Please check your stress tensor!")
            sys.exit(1)
            
        # Calculate derivatives of position & com velocity #
        self.down_dxds_data = first_derivative(base_array = self.down_position_data,
                                        deriv_size = self.down_ds,axis = 0,
                                        ar_size = self.down_position_data.shape[0], 
                                        dim = 3)
        
        
        self.down_d2xds2_data = second_derivative(base_array = self.down_position_data,
                                        deriv_size = self.down_ds,axis = 0,
                                        ar_size = self.down_position_data.shape[0], 
                                        dim = 3)
        
        self.down_ducomdt_data = first_derivative(base_array = self.down_com_data,
                                        deriv_size = self.down_dt,axis = 0,
                                        ar_size = self.down_com_data.shape[0], 
                                        dim = 2)
        
        self.down_d2ucomdt2_data = second_derivative(base_array = self.down_com_data,
                                        deriv_size = self.down_dt,axis = 0,
                                        ar_size = self.down_com_data.shape[0], 
                                        dim = 2)
        
        # Calculate Maximum curvature along filament #
        self.down_curvature_data = (np.abs((self.down_dxds_data[:,0,:]*self.down_d2xds2_data[:,1,:]) -\
                                         (self.down_dxds_data[:,1,:]*self.down_d2xds2_data[:,0,:])))/\
            (((self.down_dxds_data[:,0,:]**2)+(self.down_dxds_data[:,1,:]**2))**(1.5))
            
        self.down_max_curv_data = self.down_curvature_data.max(axis = 0)
        
        #Calculate angle of orientation of filament
        self.down_angle_orientation = np.rad2deg(np.mod(np.arctan2(self.down_dxds_data[:,1,:],self.down_dxds_data[:,0,:]),2*np.pi))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_dir", 
                        help="Specify path to the directory where the relevant scripts are located in",
                    type = str)
    parser.add_argument("input_up_dir", 
                        help="Specify path to the directory where the upward flip data files are located in",
                    type = str)
    parser.add_argument("input_down_dir", 
                        help="Specify path to the directory where the downward flip data files are located in",
                    type = str)
    parser.add_argument("output_dir", 
                        help="Specify path to the directory where the output files will be saved in",
                    type = str)
    
    #Uncommment this line when debugging or running from a terminal
    args = parser.parse_args(['C:\\Users\\super\\OneDrive - University of California, Davis\\School\\UCD_Files\\Work\\00_Projects\\02_Shear_Migration\\00_Scripts\\01c_Non_Brownian_Snap_LP\\',
                              'C:\\Users\\super\\OneDrive - University of California, Davis\\School\\UCD_Files\\Work\\00_Projects\\02_Shear_Migration\\00_Scripts\\01_Migration_Simulations\\02_Actual_Results\\NB_Poiseuille\\UC_1p00_MB_1p00e5\\Up_0p25\\',
                              'C:\\Users\\super\\OneDrive - University of California, Davis\\School\\UCD_Files\\Work\\00_Projects\\02_Shear_Migration\\00_Scripts\\01_Migration_Simulations\\02_Actual_Results\\NB_Poiseuille\\UC_1p00_MB_1p00e5\\Down_0p25\\',
                              'C:\\Users\\super\\OneDrive - University of California, Davis\\School\\UCD_Files\\Work\\00_Projects\\02_Shear_Migration\\00_Scripts\\01c_Non_Brownian_Snap_LP\\01_Actual_Results\\Up_v_Down_0p25\\']) 
    # args = parser.parse_args()
    
    os.chdir(args.project_dir)

    from misc.create_dir import create_dir
    from calculations.first_derivative import first_derivative
    from calculations.second_derivative import second_derivative
    from calculations.center_of_mass import center_of_mass
    from plotting.plot_panels import plot_time_course_panels

    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s|%(filename)s|%(levelname)s|%(message)s',
            datefmt="%A, %B %d at %I:%M:%S %p")
    logging.info(
        "Started reading in data files.")
    
    comp_data = up_down_data(up_dir = args.input_up_dir,
                             down_dir = args.input_down_dir,
                             output_dir = args.output_dir)
    logging.info(
        "Finished reading in data files.")
    
    ### Plotting data ###
    
    # #Position over time
    # plot_time_course_panels(x_values_up = comp_data.up_position_data[:,0,:],
    #                         x_values_down = comp_data.down_position_data[:,0,:],
    #                         y_values_up = comp_data.up_position_data[:,1,:],
    #                         y_values_down = comp_data.down_position_data[:,1,:],
    #                         timepoints_up = comp_data.up_time_data_adj,
    #                         timepoints_down = comp_data.down_time_data_adj,
    #                         xlabels = r"$x$",ylabels = r"$y$",
    #                         plot_type = 'position',output_dir = comp_data.output_dir,
    #                         file_name = 'all_filament_position')
    
    # #Tension over Time
    # plot_time_course_panels(x_values_up = comp_data.s_up,
    #                         x_values_down = comp_data.s_down,
    #                         y_values_up = comp_data.up_tension_data,
    #                         y_values_down = comp_data.down_tension_data,
    #                         timepoints_up = comp_data.up_time_data_adj,
    #                         timepoints_down = comp_data.down_time_data_adj,
    #                         xlabels = r"$s$",ylabels = r"$T(s)$",
    #                         plot_type = 'tension',output_dir = comp_data.output_dir,
    #                         file_name = 'all_filament_tension')
    
    # #Elastic Energy Over Time
    # plot_time_course_panels(x_values_up = comp_data.up_time_data_adj,
    #                         x_values_down = comp_data.down_time_data_adj,
    #                         y_values_up = comp_data.up_elastic_data,
    #                         y_values_down = comp_data.down_elastic_data,
    #                         timepoints_up = comp_data.up_time_data_adj,
    #                         timepoints_down = comp_data.down_time_data_adj,
    #                         xlabels = r"$t/t_{\text{max}}$",ylabels = r"$E_{\text{elastic}}$",
    #                         plot_type = 'elastic_energy_time',output_dir = comp_data.output_dir,
    #                         file_name = 'all_filament_elastic_energy')
    
    # #N1 Stress Over Time
    # plot_time_course_panels(x_values_up = comp_data.up_time_data_adj,
    #                         x_values_down = comp_data.down_time_data_adj,
    #                         y_values_up = comp_data.up_N1_data,
    #                         y_values_down = comp_data.down_N1_data,
    #                         timepoints_up = comp_data.up_time_data_adj,
    #                         timepoints_down = comp_data.down_time_data_adj,
    #                         xlabels = r"$t/t_{\text{max}}$",ylabels = r"$N_{1}$",
    #                         plot_type = 'N1_Stress_time',output_dir = comp_data.output_dir,
    #                         file_name = 'all_filament_N1')
    
    # #N2 Stress Over Time
    # plot_time_course_panels(x_values_up = comp_data.up_time_data_adj,
    #                         x_values_down = comp_data.down_time_data_adj,
    #                         y_values_up = comp_data.up_N2_data,
    #                         y_values_down = comp_data.down_N2_data,
    #                         timepoints_up = comp_data.up_time_data_adj,
    #                         timepoints_down = comp_data.down_time_data_adj,
    #                         xlabels = r"$t/t_{\text{max}}$",ylabels = r"$N_{2}$",
    #                         plot_type = 'N2_Stress_time',output_dir = comp_data.output_dir,
    #                         file_name = 'all_filament_N2')
    
    # #Sxy Stress Over Time
    # plot_time_course_panels(x_values_up = comp_data.up_time_data_adj,
    #                         x_values_down = comp_data.down_time_data_adj,
    #                         y_values_up = comp_data.up_Sxy_data,
    #                         y_values_down = comp_data.down_Sxy_data,
    #                         timepoints_up = comp_data.up_time_data_adj,
    #                         timepoints_down = comp_data.down_time_data_adj,
    #                         xlabels = r"$t/t_{\text{max}}$",ylabels = r"$\sigma_{xy}$",
    #                         plot_type = 'Sxy_Stress_time',output_dir = comp_data.output_dir,
    #                         file_name = 'all_filament_Sxy')
    
    
#%% Elastic Energy & Max curvature
end_time = 20
# up_end_time_idx = np.where(comp_data.up_time_data == 20)[0][0] + 1
# down_end_time_idx = np.where(comp_data.down_time_data == 20)[0][0] + 1

# fig,axes = plt.subplots(nrows = 2,ncols = 2,figsize = (7,7),sharey = True,sharex = True,layout = 'constrained')
# axes[0,0].plot(comp_data.up_time_data[:up_end_time_idx],comp_data.up_elastic_data[:up_end_time_idx],'r',label = 'Upward Flip')
# axes[0,0].plot(comp_data.down_time_data[:down_end_time_idx],comp_data.down_elastic_data[:down_end_time_idx],'b',label = 'Downward Flip')
# axes[0,1].plot(comp_data.up_time_data[:up_end_time_idx],comp_data.up_elastic_data[:up_end_time_idx],'r',label = 'Upward Flip')
# axes[0,1].plot(comp_data.down_time_data[:down_end_time_idx]-1.75,comp_data.down_elastic_data[:down_end_time_idx],'b',label = 'Downward Flip (shifted)')

# axes[1,0].plot(comp_data.up_time_data[:up_end_time_idx],comp_data.up_max_curv_data[:up_end_time_idx],'r',label = 'Upward Flip')
# axes[1,0].plot(comp_data.down_time_data[:down_end_time_idx],comp_data.down_max_curv_data[:down_end_time_idx],'b',label = 'Downward Flip')
# axes[1,1].plot(comp_data.up_time_data[:up_end_time_idx],comp_data.up_max_curv_data[:up_end_time_idx],'r',label = 'Upward Flip')
# axes[1,1].plot(comp_data.down_time_data[:down_end_time_idx]-1.6,comp_data.down_max_curv_data[:down_end_time_idx],'b',label = 'Downward Flip (shifted)')

# for n_row,ax_row in enumerate(axes):
#     for n_col,ax_col in enumerate(ax_row):
#         ax_col.set_ylim(-0.5,23)
#         ax_col.set_xlim(-0.75,21)
#         ax_col.set_xticks(np.linspace(0,20,5))
#         ax_col.set_yticks(np.linspace(0,20,5))
#         ax_col.tick_params(axis = 'both',which = 'both',labelsize= 11,direction = 'in')
#         ax_col.legend(loc = 'upper right')
#         ax_col.set_aspect(np.diff(ax_col.get_xlim())/np.diff(ax_col.get_ylim()))

# axes[0,0].set_ylabel(r"$E_{\text{elastic}}$",fontsize = 13)
# axes[1,0].set_ylabel(r"$\text{max}(k)$",fontsize = 13)
# fig.supxlabel(r"$t$",y = -0.03,size = 13)
# plt.savefig(os.path.join(comp_data.output_dir,'elastic_curvature_comp.png'),dpi = 400,
#             bbox_inches = 'tight')
# plt.show()


#%% Find signature of center of mass change

end_time = 20
up_end_time_idx = np.where(comp_data.up_time_data == 20)[0][0] + 1
down_end_time_idx = np.where(comp_data.down_time_data == 20)[0][0] + 1

fig,axes = plt.subplots(figsize = (7,7),layout = 'constrained')

axes.plot(comp_data.up_time_data[:up_end_time_idx],comp_data.up_com_data[:up_end_time_idx,1],'r',label = 'Center of Mass')
axes.set_ylim(0,0.041)
ax1 = axes.twinx()
ax1.plot(comp_data.up_time_data[:up_end_time_idx],comp_data.up_dxds_data[:,1,:].max(axis = 0)[:up_end_time_idx],'b')   
ax1.vlines(x = 5.35,ymin = 0,ymax = 1.1,color = 'black',alpha = 0.7,linestyle = 'dashed')
ax1.vlines(x = 10.5,ymin = 0,ymax = 1.1,color = 'black',alpha = 0.7,linestyle = 'dashed')
ax1.set_ylim(0,1.01)
plt.show()     
        
        
