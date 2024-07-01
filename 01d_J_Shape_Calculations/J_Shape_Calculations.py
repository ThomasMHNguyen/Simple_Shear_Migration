# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:07:07 2023

@author: super
"""
import re, sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

input_dir = 'C:\\Users\\super\\OneDrive - University of California, Davis\\School\\UCD_Files\\Work\\00_Projects\\02_Shear_Migration\\00_Scripts\\01d_J_Shape_Calculations\\\\'
os.chdir(input_dir)
output_dir = os.path.join(input_dir,'Plots/')
from create_dir import create_dir

create_dir(output_dir)
class filament_parameters():
    def __init__(self,N,theta_start,flow_type,flip_type):
        self.N = N
        self.s = np.linspace(-1/2,1/2,self.N)
        self.ds = 1/(self.N-1)
        self.theta_start = theta_start
        self.fil_end_theta = theta_start + np.pi
        self.s_arc_N = 30
        self.s_arm_N = self.N - self.s_arc_N
        self.s_arc = self.s[-self.s_arc_N:]
        self.s_arm = self.s[:-self.s_arc_N]
        self.arc_radius = self.s_arc[-1] - self.s_arc[0]
        self.s_arc_loc = np.linspace(self.theta_start,self.fil_end_theta,self.s_arc_N)
        self.s_arc_multp = self.s_arc_loc/self.s_arc
        self.flow_type = flow_type
        self.flip_type = flip_type
        self.U_centerline = 1
        self.channel_height = 0.5

    def first_deriv(self,base_array):
        """
        This function calculates the first derivative of the scalar or matrix of 
        interest and applies the appropriate end correction terms accurate to O(s^2).
        
        Inputs:
            
        base_array:     Nx3 array who needs its first derivative to be calculated.
        N:              Number of points used to discretize the length of the filament.
        ds:             spacing between each point on the filament.
        dim:            The number of dimensions the array has.
        """
        ar_size = base_array.shape[0]
        self.dds_base = np.zeros((ar_size,3),dtype = float)
        self.dds_base[0,:] = (0.5/self.ds)*(-3*base_array[0,:] + 4*base_array[1,:] - 1*base_array[2,:])
        self.dds_base[ar_size-1,:] = (0.5/self.ds)*(3*base_array[ar_size-1,:] - 4*base_array[ar_size-2,:] + 1*base_array[ar_size-3,:])
        self.dds_base[1:ar_size-1,:] = (0.5/self.ds)*(-base_array[0:ar_size-2,:] + base_array[2:ar_size,:])

        return self.dds_base
    
    def configure_coord(self):
        if self.flow_type == 'Poiseuille':
            if self.flip_type == 'Up':
                
                ### Calculate coordinates of arc ###
                self.xarc =  1/np.pi*(self.arc_radius)*-np.cos(self.s_arc_loc)
                self.yarc = 1/np.pi*(self.arc_radius)*np.sin(self.s_arc_loc)
                self.zarc = np.zeros(self.s_arc_loc.shape[0],dtype = float)
                self.arcloc = np.column_stack((self.xarc,self.yarc,self.zarc))
                
                ### Calculate coordinates of arm ###
                self.dxds_arc = self.first_deriv(self.arcloc)
                self.arc_end_angle = np.arctan(self.dxds_arc[0,1]/self.dxds_arc[0,0])
                self.arm_slope = self.dxds_arc[0,1]/self.dxds_arc[0,0]
                self.arm_intercept = (-self.arm_slope*self.arcloc[0,0]) + self.arcloc[0,1]
                
                self.xarm = self.s_arm*np.cos(self.arc_end_angle)
                self.yarm = self.s_arm*np.sin(self.arc_end_angle)
                self.zarm = np.zeros(self.s_arm.shape[0],dtype = float)
                self.armloc = np.column_stack((self.xarm,self.yarm,self.zarm))
                
                ### Adjust x-component of arm ###
                self.arm_dist =  self.arcloc[0,:] - self.armloc[-1,:] - (self.armloc[-1,:] - self.armloc[-2,:])
                self.armloc = self.armloc  + self.arm_dist
                self.x_loc = np.concatenate((self.armloc,self.arcloc),axis = 0)
                
            if self.flip_type == 'Down':
                """
                Note that for the downward flips, the coordinates of the arc 
                and arm components need to be flipped.
                """
                
                ### Calculate coordinates of arc ###
                self.xarc =  1/np.pi*(self.arc_radius)*-np.cos(self.s_arc_loc)[::-1]
                self.yarc = 1/np.pi*(self.arc_radius)*-np.sin(self.s_arc_loc)[::-1]
                self.zarc = np.zeros(self.s_arc_loc.shape[0],dtype = float)
                self.arcloc = np.column_stack((self.xarc,self.yarc,self.zarc))
                
                ### Calculate coordinates of arm ###
                self.dxds_arc = self.first_deriv(self.arcloc)
                self.arc_end_angle = np.arctan(self.dxds_arc[0,1]/self.dxds_arc[0,0])
                self.arm_slope = self.dxds_arc[0,1]/self.dxds_arc[0,0]
                self.arm_intercept = (-self.arm_slope*self.arcloc[0,0]) + self.arcloc[0,1]
                
                self.xarm = self.s_arm[::-1]*np.cos(self.arc_end_angle)
                self.yarm = self.s_arm[::-1]*np.sin(self.arc_end_angle)
                self.zarm = np.zeros(self.s_arm.shape[0],dtype = float)
                self.armloc = np.column_stack((self.xarm,self.yarm,self.zarm))
                
                ### Adjust x-component of arm ###
                self.arm_dist =  self.arcloc[0,:] - self.armloc[-1,:] - (self.armloc[-1,:] - self.armloc[-2,:])
                self.armloc = self.armloc  + self.arm_dist
                self.x_loc = np.concatenate((self.armloc,self.arcloc),axis = 0)       
                
    def adjust_arc_end_coord(self):
        self.dot_theta = np.arccos(np.dot(self.x_loc[-1,:],self.x_loc[-2,:])/\
                                   (np.linalg.norm(self.x_loc[-1,:]) * np.linalg.norm(self.x_loc[-2,:])))
            
        self.new_end_loc = (self.x_loc[-1,:]*np.cos(self.dot_theta)).reshape(1,3)
        self.x_loc[-1,:] = self.new_end_loc
        
    
    def calculate_filament_fluid_velocity(self):
        self.U0 = np.zeros(shape = (self.N,3),dtype = float)
        self.U0[:,0] = self.U_centerline*(1-(self.x_loc[:,1]**2/self.channel_height**2)) 
     
    def find_filament_length(self):
        self.filament_length = np.sqrt(((self.x_loc[1:self.N,:]-self.x_loc[0:self.N-1,:])**2).sum(axis = 1)).sum()
        
    def calculate_filament_components_velocity(self):
        self.avg_arm_velocity = self.U0[:-self.s_arc_N,0].mean()
        self.avg_arc_velocity = self.U0[-self.s_arc_N:,0].mean()
        self.avg_fil_velocity = np.trapz(self.U0[:,0])
        
    def calculate_center_of_mass(self):
        self.center_mass = self.x_loc.mean(axis = 0)

#%% Poiseuille-Up 1 instance

j_shape_filament = filament_parameters(N = 101,
                                       theta_start = 3*np.pi/4,
                                       flow_type = 'Poiseuille',
                                       flip_type = 'Up')

j_shape_filament.configure_coord()

fig,axes = plt.subplots(ncols = 2,figsize = (7,7))
axes[0].plot(j_shape_filament.arcloc[:,0],j_shape_filament.arcloc[:,1],'r')
axes[0].plot(j_shape_filament.armloc[:,0],j_shape_filament.armloc[:,1],'b')
axes[0].plot(j_shape_filament.arcloc[0,0],j_shape_filament.arcloc[0,1],'cyan',marker = 'd',
          markersize = 1)
axes[0].plot(j_shape_filament.armloc[-1,0],j_shape_filament.armloc[-1,1],'magenta',marker = 'o',
          markersize = 1)
axes[1].plot(j_shape_filament.x_loc[:,0],j_shape_filament.x_loc[:,1],color = 'gray')

for ax in axes:
    ax.set_xlim(-0.6,0.6)
    ax.set_ylim(-0.6,0.6)
    ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0])/(ax.get_ylim()[1] - ax.get_ylim()[0]))
plt.show()


#%% Poiseuille-Up Cycle through theta values

theta_vals = np.linspace(101*np.pi/200,199*np.pi/200,100)
filament_avg_velocity_x = np.zeros(theta_vals.shape[0],dtype = float)
filament_com_y = np.zeros(theta_vals.shape[0],dtype = float)
arm_orient = np.zeros(theta_vals.shape[0],dtype = float)
for i,theta in enumerate(theta_vals):
    j_shape_filament = filament_parameters(N = 101,
                                           theta_start = theta,
                                           flow_type = 'Poiseuille',
                                           flip_type = 'Up')
    j_shape_filament.configure_coord()
    # j_shape_filament.find_filament_length()
    # print(j_shape_filament.filament_length)
    j_shape_filament.calculate_filament_fluid_velocity()
    j_shape_filament.calculate_filament_components_velocity()
    j_shape_filament.calculate_center_of_mass()
    filament_avg_velocity_x[i]=  j_shape_filament.avg_fil_velocity
    filament_com_y[i]=  j_shape_filament.center_mass[1]
    arm_orient[i] = 180 + np.degrees(j_shape_filament.arc_end_angle)
    
fig,axes = plt.subplots(figsize = (7,7))
axes.plot(arm_orient,filament_com_y,'r',label = 'COM-y')
axes_r = axes.twinx()
axes_r.plot(arm_orient,filament_avg_velocity_x,'b',label = 'Avg Filament Velocity')
axes.set_aspect((axes.get_xlim()[1] - axes.get_xlim()[0])/(axes.get_ylim()[1] - axes.get_ylim()[0]))
axes_r.set_aspect((axes_r.get_xlim()[1] - axes_r.get_xlim()[0])/(axes_r.get_ylim()[1] - axes_r.get_ylim()[0]))
axes.set_xlim(85,185)
axes.set_xticks(np.linspace(90,180,7))
axes.set_title("Poiseuille-Upward Flips")
fig.legend()
axes.set_ylabel("COM-y")
axes.set_xlabel(r"$\theta_{f}\: [\deg]$")
axes_r.set_ylabel("Average Filament Velocity")
plt.savefig(os.path.join(output_dir,'Up_Poiseuille_COM_Fil_Velocity.png'),dpi = 200,bbox_inches = 'tight')
plt.show()

#%% Poiseuille-Down 1 instance

j_shape_filament = filament_parameters(N = 101,
                                       theta_start = 225*np.pi/200,
                                       flow_type = 'Poiseuille',
                                       flip_type = 'Down')
j_shape_filament.configure_coord()
j_shape_filament.find_filament_length()
# print(j_shape_filament.filament_length)
j_shape_filament.calculate_filament_fluid_velocity()

fig,axes = plt.subplots(ncols = 2,figsize = (7,7))
axes[0].plot(j_shape_filament.arcloc[:,0],j_shape_filament.arcloc[:,1],'r')
axes[0].plot(j_shape_filament.armloc[:,0],j_shape_filament.armloc[:,1],'b')
axes[0].plot(j_shape_filament.arcloc[0,0],j_shape_filament.arcloc[0,1],'cyan',marker = 'd',
         markersize = 3)
axes[0].plot(j_shape_filament.armloc[-1,0],j_shape_filament.armloc[-1,1],'magenta',marker = 'o',
         markersize = 3)
axes[1].plot(j_shape_filament.x_loc[:,0],j_shape_filament.x_loc[:,1],color = 'gray')

for ax in axes:
    ax.set_xlim(-0.6,0.6)
    ax.set_ylim(-0.6,0.6)
    ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0])/(ax.get_ylim()[1] - ax.get_ylim()[0]))
plt.show()

#%% Poiseuille-Down Cycle through theta values

theta_vals = np.linspace(201*np.pi/200,299*np.pi/200,100)
filament_avg_velocity_x = np.zeros(theta_vals.shape[0],dtype = float)
filament_com_y = np.zeros(theta_vals.shape[0],dtype = float)
arm_orient = np.zeros(theta_vals.shape[0],dtype = float)
for i,theta in enumerate(theta_vals):
    j_shape_filament = filament_parameters(N = 101,
                                           theta_start = theta,
                                           flow_type = 'Poiseuille',
                                           flip_type = 'Down')
    j_shape_filament.configure_coord()
    # j_shape_filament.find_filament_length()
    # print(j_shape_filament.filament_length)
    j_shape_filament.calculate_filament_fluid_velocity()
    j_shape_filament.calculate_filament_components_velocity()
    j_shape_filament.calculate_center_of_mass()
    filament_avg_velocity_x[i]=  j_shape_filament.avg_fil_velocity
    filament_com_y[i]=  j_shape_filament.center_mass[1]
    arm_orient[i] = 360 + np.degrees(j_shape_filament.arc_end_angle)
    
fig,axes = plt.subplots(figsize = (7,7))
axes.plot(arm_orient,filament_com_y,'r',label = 'COM-y')
axes_r = axes.twinx()
axes_r.plot(arm_orient,filament_avg_velocity_x,'b',label = 'Avg Filament Velocity')
axes.set_aspect((axes.get_xlim()[1] - axes.get_xlim()[0])/(axes.get_ylim()[1] - axes.get_ylim()[0]))
axes_r.set_aspect((axes_r.get_xlim()[1] - axes_r.get_xlim()[0])/(axes_r.get_ylim()[1] - axes_r.get_ylim()[0]))
axes.set_xlim(265,365)
axes.set_xticks(np.linspace(270,360,7))
axes.set_title("Poiseuille-Downward Flips")
fig.legend()
axes.set_ylabel("COM-y")
axes.set_xlabel(r"$\theta_{f}\: [\deg]$")
axes_r.set_ylabel("Average Filament Velocity")
plt.savefig(os.path.join(output_dir,'Down_Poiseuille_COM_Fil_Velocity.png'),dpi = 200,bbox_inches = 'tight')
plt.show()


#%% Poiseuille-Up or Down
### Down
# theta_test = np.linspace(5*np.pi/4,5*np.pi/4 + np.pi,100)
# x = -1*np.cos(theta_test)[::-1]
# y = -1*np.sin(theta_test)[::-1]

### Up 
theta_test = np.linspace(3*np.pi/4,3*np.pi/4 + np.pi,100)
x = -1*np.cos(theta_test)
y = 1*np.sin(theta_test)

z = 1*np.zeros(theta_test.shape[0])
circ_loc = np.column_stack((x,y,z))

s1_deriv = np.zeros(shape = (100,3),dtype = float)
s1_deriv[0,:] = (0.5/(1/100-1))*(-3*circ_loc[0,:] + 4*circ_loc[1,:] - 1*circ_loc[2,:])
s1_deriv[100-1,:] = (0.5/(1/100-1))*(3*circ_loc[100-1,:] - 4*circ_loc[100-2,:] + 1*circ_loc[100-3,:])
s1_deriv[1:100-1,:] = (0.5/(1/100-1))*(-circ_loc[0:100-2,:] + circ_loc[2:100,:])


theta = np.arccos(np.dot(circ_loc[-1,:],circ_loc[-2,:])/(np.linalg.norm(circ_loc[-1,:]) * np.linalg.norm(circ_loc[-2,:])))
new_loc = (circ_loc[-1,:]*np.cos(theta)).reshape(1,3)
test_vector = np.concatenate((new_loc,circ_loc[-2,:].reshape(1,3)),axis = 0)
plt.plot(circ_loc[:,0],circ_loc[:,1],'r')
plt.plot(np.array([0,test_vector[1,0]]),np.array([0,test_vector[1,1]]),'b')
plt.plot(np.array([0,test_vector[0,0]]),np.array([0,test_vector[0,1]]),'g')
plt.gca().set_aspect(1)
plt.show()
