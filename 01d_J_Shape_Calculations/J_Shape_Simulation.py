# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:02:42 2023

@author: super
"""
import re, sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special, sparse
import pandas as pd

os.chdir('C://Users//super//OneDrive - University of California, Davis//School//UCD_Files//Work//00_Projects//02_Shear_Migration//00_Scripts//01d_J_Shape_Calculations//')
from create_dir import create_dir
#%% Functions
def first_deriv(base_array,N,ds,dim):
    """
    This function calculates the first derivative of the scalar or matrix of 
    interest and applies the appropriate end correction terms accurate to O(s^2).
    
    Inputs:
        
    base_array:     Nx3 array who needs its first derivative to be calculated.
    N:              Number of points used to discretize the length of the filament.
    ds:             spacing between each point on the filament.
    dim:            The number of dimensions the array has.
    """
    if dim == 1:
        dds_base = np.zeros((N),dtype = float)
        dds_base[0] = (0.5/ds)*(-3*base_array[0] + 4*base_array[1] - 1*base_array[2])
        dds_base[N-1] = (0.5/ds)*(3*base_array[N-1] - 4*base_array[N-2] + 1*base_array[N-3])
        dds_base[1:N-1] = (0.5/ds)*(-base_array[0:N-2] + base_array[2:N])
    elif dim == 2:
        dds_base = np.zeros((N,3),dtype = float)
        dds_base[0,:] = (0.5/ds)*(-3*base_array[0,:] + 4*base_array[1,:] - 1*base_array[2,:])
        dds_base[N-1,:] = (0.5/ds)*(3*base_array[N-1,:] - 4*base_array[N-2,:] + 1*base_array[N-3,:])
        dds_base[1:N-1,:] = (0.5/ds)*(-base_array[0:N-2,:] + base_array[2:N,:])
    return dds_base


def second_deriv(base_array,N,ds):
    """
    This function calculates the 2nd derivative of a vector and applies the 
    appropirate end correction terms accurate to 0(s^2).
    
    Inputs:
        
    base_array:     Nx3 array who needs its first derivative to be calculated.
    N:              Number of points used to discretize the length of the filament.
    ds:             spacing between each point on the filament.
    """
    d2ds2_base = np.zeros((N,3),dtype = float)
    d2ds2_base[0,:] = (1/ds**2)*(2*base_array[0,:]-5*base_array[1,:]+4*base_array[2,:]-base_array[3,:])
    d2ds2_base[N-1,:] = (1/ds**2)*(2*base_array[N-1,:]-5*base_array[N-2,:]+4*base_array[N-3,:]-base_array[N-4,:])
    d2ds2_base[1:N-1,:] = (1/ds**2)*(base_array[2:N,:]-2*base_array[1:N-1,:]+base_array[0:N-2,:])
    return d2ds2_base


def third_deriv(base_array,N,ds):
    """
    This function calculates the 3rd derivative of a vector and applies the 
    appropirate end correction terms accurate to 0(s^2). Additional end corrections
    are applied via https://www.geometrictools.com/Documentation/FiniteDifferences.pdf. 
    
    Inputs:
        
    base_array:     Nx3 array who needs its first derivative to be calculated.
    N:              Number of points used to discretize the length of the filament.
    ds:             spacing between each point on the filament.
    """
    d3ds3_base = np.zeros((N,3),dtype = float)
    d3ds3_base[0,:] = 1/ds**3*(-2.5*base_array[0,:]+9*base_array[1,:]-12*base_array[2,:]+7*base_array[3,:]-1.5*base_array[4,:])
    d3ds3_base[1,:] = 1/ds**3*(-1.5*base_array[0,:]+5*base_array[1,:]-6*base_array[2,:]+3*base_array[3,:]-0.5*base_array[4,:])             
    d3ds3_base[N-1,:] = 1/ds**3*(2.5*base_array[N-1,:]-9*base_array[N-2,:]+12*base_array[N-3,:]-7*base_array[N-4,:]+1.5*base_array[N-5,:])
    d3ds3_base[N-2,:] = 1/ds**3*(1.5*base_array[N-1,:]-5*base_array[N-2,:]+6*base_array[N-3,:]-3*base_array[N-4,:]+0.5*base_array[N-5,:])     
    d3ds3_base[2:N-2,:] = 1/ds**3*(0.5*base_array[4:N,:]-base_array[3:N-1,:]+base_array[1:N-3,:]-0.5*base_array[0:N-4,:])
    return d3ds3_base
 
       
def fourth_deriv(base_array,N,ds):
    """
    This function calculates the 4th derivative of a vector and applies the 
    appropirate end correction terms accurate to 0(s^2). Additional end corrections
    are applied via https://www.geometrictools.com/Documentation/FiniteDifferences.pdf. 
    
    Inputs:
        
    base_array:     Nx3 array who needs its first derivative to be calculated.
    N:              Number of points used to discretize the length of the filament.
    ds:             spacing between each point on the filament.
    """
    d4ds4_base = np.zeros((N,3),dtype = float)
    d4ds4_base[0,:] = 1/ds**4*(3*base_array[0,:]-14*base_array[1,:]+26*base_array[2,:]-24*base_array[3,:]+11*base_array[4,:]-2*base_array[5,:])
    d4ds4_base[1,:] = 1/ds**4*(2*base_array[0,:]-9*base_array[1,:]+16*base_array[2,:]-14*base_array[3,:]+6*base_array[4,:]-1*base_array[5,:])        
    d4ds4_base[N-1,:] = 1/ds**4*(3*base_array[N-1,:]-14*base_array[N-2,:]+26*base_array[N-3,:]-24*base_array[N-4,:]+11*base_array[N-5,:]-2*base_array[N-6,:])
    d4ds4_base[N-2,:] = 1/ds**4*(2*base_array[N-1,:]-9*base_array[N-2,:]+16*base_array[N-3,:]-14*base_array[N-4,:]+6*base_array[N-5,:]-1*base_array[N-6,:])    
    d4ds4_base[2:N-2,:] = 1/ds**4*(base_array[4:N,:]-4*base_array[3:N-1,:]+6*base_array[2:N-2,:]-4*base_array[1:N-3,:]+base_array[0:N-4,:])
    return d4ds4_base


def spatialderiv(sim_c,x_vec):
    """
    This function evaluates the spatial derivatives of the filament.
    
    Inputs:
    sim_c:       Class that contains all parameters needed for the simulation 
                 and arrays to store data.
    x_vec:       Nx3 vectorized array that contains the location of each point
    """
    N,ds = sim_c.N,sim_c.ds
    dxds = first_deriv(x_vec,N,ds,2)
    d2xds2 = second_deriv(x_vec,N,ds)
    d3xds3 = third_deriv(x_vec,N,ds)
    d4xds4 = fourth_deriv(x_vec,N,ds) 
    return dxds, d2xds2, d3xds3, d4xds4


def rdot(ar1,ar2):
    """
    This function computes the dot product of 2 vectors (Check LaTeX markup for
     actual markup) but accounts for the vectors as numpy arrays.
    
    Inputs:
    ar1:        Numpy 2D array representing Vector #1
    ar2:        Numpy 2D array representing Vector #2
    """
    return np.sum(np.multiply(ar1,ar2),axis=1)


def fluid_velo(sim_c,x_vec):
    """
    This function calculates the fluid velocity based on the position of the 
    filament. 
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    x_vec:      Nx3  array that contains the coordinates of each point.
    t:          current iteration of simulation.
    """
    N = sim_c.N
    # s = sim_c.s
    ds = sim_c.ds
    H = sim_c.channel_upper_height
    U_centerline = sim_c.U_centerline
    # k_flow_phase,k_flow_factor = sim_c.kflow_phase,sim_c.kflow_freq
    
    u0 = np.zeros([N,3],dtype = float)
    
    ### Shear flow ###
    # u0[:,0] = x_vec[:,1]
    
    ### Poiseuille Flow for H = L ###
    u0[:,0] = U_centerline*(1-(x_vec[:,1]**2/H**2))
        
    ### Kolmogorov Flow ###
    # u0[:,0] = np.sin(k_flow_phase*k_flow_factor*x_vec[:,1])
    
    ### Velocity derivative ###
    du0ds = first_deriv(u0,N,ds,2)
    
    return u0, du0ds

def rod_length(N,x_vec):
    """
    This function calculates the length of the filament.
    
    Inputs:
    N:          Number of points used to discretize filament.
    x_vec:      Nx3 array that contains the coordinates of each point along the
                filament.
    """
    r_length = np.sqrt(((x_vec[1:N,:]-x_vec[0:N-1,:])**2).sum(axis = 1)).sum()
    return r_length

def calculate_N_ex(sim_c,params):
    """
    This function calculates the non-semi implicit terms.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    params:     list of arrays that contain components needed to solve explicit terms
                in the semi-implicit time-stepping. 
    """
    N,c, mu_bar = sim_c.N,sim_c.c, sim_c.mu_bar
    
    #Unpack params
    u0, du0ds, dxds, d2xds2, d3xds3,\
    Txss,Tsxs,Kxs,Ksxs,Ksxsss,Kssxss,brownian = params
    
    N_ex = (mu_bar*u0) - ((c+1)*(-Tsxs + Kssxss - Txss + 2*Ksxsss) + 
                          (c-3)*(-Tsxs + 2*Ksxs*np.repeat(rdot(dxds,d3xds3),3).reshape(N,3)))                                     
    return N_ex

def solve_tension(sim_c,du0ds,dxds,d2xds2,d3xds3,d4xds4):
    """
    This function solves for the tension equation using np.linalg.solve.
    
    Inputs:

    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    du0ds:      Nx3 vectorized array that represents 1st spatial derivative of 
                fluid velocity.
    dxds:       Nx3 vectorized array that represents 1st spatial derivative of 
                filament position. 
    d2xds2:     Nx3 vectorized array that represents 2nd spatial derivative of 
                filament position. 
    d3xds3:     Nx3 vectorized array that represents 3rd spatial derivative of 
                filament position. 
    d4xds4:     Nx3 vectorized array that represents 4th spatial derivative of 
                filament position. 
    brownian:   Nx3 array that represents the Brownian forces acting on each
                point on the filament.
    """
    mu_bar,N,c,ds = sim_c.mu_bar,sim_c.N,sim_c.c,sim_c.ds
    K,Ks,Kss = sim_c.K,sim_c.Ks,sim_c.Kss
       
    # Evaluating Tension Equation with BC: Tension = 0 at ends of Filament 
    a = np.ones(N-3)*(-2*(c-1)/ds**2) # Lower and Upper Diag
    b = np.ones(N-2)*(4*(c-1)/ds**2)+ (c+1)*rdot(d2xds2,d2xds2)[1:N-1] # Center Diag  
    d = ((mu_bar*rdot(du0ds,dxds))+ ((5*c-3)*(Kss*rdot(d2xds2,d2xds2))) + \
          (4*(4*c-3)*(Ks*rdot(d2xds2,d3xds3))) + \
            ((7*c-5)*K*rdot(d2xds2,d4xds4))+(6*(c-1)*K*rdot(d3xds3,d3xds3)) -\
          sim_c.zeta_use*(1-rdot(dxds,dxds)))[1:N-1] # RHS-non constant K
        
    ### Evluate tension ###
    A = sparse.diags([a,b,a],offsets = [-1,0,1],shape = (N-2,N-2)).toarray()
    tension = np.insert(np.linalg.solve(A,d),(0,N-2),0)
    
    return tension


def calc_force(sim_c,Txss,Tsxs,Kssxss,Ksxsss,Kxssss,f_type):
    """
    This function calculates the force experienced by the filament due to tensile
    and rigidity forces.
    
    Inputs:

    Txss:       Nx3 vectorized array of tension multiplied by 2nd derivative 
                of filament position. 
    Tsxs:       Nx3 vectorized array of tension derivative multplied by 1st 
                derivative of filament position. 
    Kssxss:     Nx3 vectorized array of 2nd derivative of rigidity multiplied 
                by 2nd derivative of filament position. 
    Ksxsss:     Nx3 vectorized array of 1st derivative of rigidity multplied 
                by 3rd derivative of filament position. 
    Kxssss:     Nx3 vectorized array of rigidity multiplied by 4th derivative 
                of filament position. 
    brownian:   Nx3 array that represents the Brownian forces acting on each
                point on the filament.
    f_type:     string argument to determine whether to use non-tensile forces
                (for sake of future non-local operator calculations) or all 
                forces (including tensile forces).
    """
    
    if f_type == 'rigid':
        force = Kssxss + 2*Ksxsss + Kxssss
    elif f_type == 'all':
        force = -Tsxs - Txss + Kssxss + 2*Ksxsss + Kxssss
    return force


def calc_vals(sim_c,tension, dTds, dxds, d2xds2, d3xds3, d4xds4):
    """
    This function creates the arrays for the terms coupled to the tension, its
    derivative, rigidity, and spatial derivatives.
    
    Inputs:

    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    tension:    Numpy 1D array that represents tension at each point on the 
                filament. 
    dTds:       Numpy 1D array that represents 1st spatial derivative of 
                tension at each point on the filament. 
    dxds:       Nx3 vectorized array that represents 1st spatial derivative of
                filament position. 
    d2xds2:     Nx3 vectorized array that represents 2nd spatial derivative of 
                filament position. 
    d3xds3:     Nx3 vectorized array that represents 3rd spatial derivative of
                filament position. 
    d4xds4:     Nx3 vectorized array that represents 4th spatial derivative of 
                filament position. 
    """
    K,Ks,Kss = sim_c.K,sim_c.Ks,sim_c.Kss
    
    Txss = d2xds2*(np.column_stack((tension,tension,tension)))
    Tsxs = dxds*(np.column_stack((dTds,dTds,dTds)))
    Kxs = dxds*(np.column_stack((K,K,K)))
    Kxssss = d4xds4*(np.column_stack((K,K,K)))
    Ksxs = dxds*(np.column_stack((Ks,Ks,Ks)))
    Ksxsss = d3xds3*(np.column_stack((Ks,Ks,Ks)))
    Kssxss = d2xds2*(np.column_stack((Kss,Kss,Kss)))
    
        
    calc_params = [Txss,Tsxs,Kxs,Kxssss,Ksxs,Ksxsss,Kssxss]
    return calc_params

def construct_lhs_matrix(sim_c,adj_xs):
    """
    This function creates the LHS matrix needed for np.linalg.solve. First, the 
    diagonals of the matrix are constructed using the finite difference coefficients.
    Next, the non-end terms (terms used for boundary conditions) are calculated by first 
    calculating the dyadic term, and then computing the dot product. After this 
    computation, the values are re-substituted back into each submatrix. Finally,
    the end terms are adjusted for the force-free and torque-free boundary conditions.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    adj_xs:     The 2D vectorized array that represents 2N(x^n_s) - N(x^n-1_s)
    """
    N, c,r1, K,Ks = sim_c.N, sim_c.c,sim_c.r1_use, sim_c.K,sim_c.Ks
    #Construct diagonals based on finite differences
    alpha = 1.5*np.ones(3*N,dtype = float)
    lhs_matrix = sparse.diags(alpha,0).toarray()
    prefactors = [r1,-4*r1,6*r1,-4*r1,r1]
    block_pos = [-6,-3,0,3,6]
    
    #### Calculate non-end terms of dyadic ###########
    identity = np.identity(3,dtype = float)
    for i in range(2,N-2):
        dyad_p = K[i]*((c+1)*identity + (c-3)*np.outer(adj_xs[i,:],adj_xs[i,:]))
        for j in block_pos: #Iterate through each index corresponding to the sub-block from the center sub-block
            lhs_matrix[3*i:3*i+3,3*i+j:3*i+j+3] += prefactors[block_pos.index(j)]*dyad_p
    
    #### Adjust end-terms for BC's ########
    lhs_matrix[0,0],lhs_matrix[1,1],lhs_matrix[2,2] = 2*K[0]*np.ones(3)
    lhs_matrix[0,3],lhs_matrix[1,4],lhs_matrix[2,5] = -5*K[0]*np.ones(3)
    lhs_matrix[0,6],lhs_matrix[1,7],lhs_matrix[2,8] = 4*K[0]*np.ones(3)
    lhs_matrix[0,9],lhs_matrix[1,10],lhs_matrix[2,11] = -1*K[0]*np.ones(3)
    
    ### 2nd row FD
    lhs_matrix[3,0],lhs_matrix[4,1],lhs_matrix[5,2] = (-float(5)/2*K[0]*np.ones(3)) + (2*Ks[0]*np.ones(3))
    lhs_matrix[3,3],lhs_matrix[4,4],lhs_matrix[5,5] = (9*K[0]*np.ones(3)) + (-5*Ks[0]*np.ones(3))
    lhs_matrix[3,6],lhs_matrix[4,7],lhs_matrix[5,8] = (-12*K[0]*np.ones(3)) + (4*Ks[0]*np.ones(3))
    lhs_matrix[3,9],lhs_matrix[4,10],lhs_matrix[5,11] = (7*K[0]*np.ones(3)) + (-1*Ks[0]*np.ones(3))
    lhs_matrix[3,12],lhs_matrix[4,13],lhs_matrix[5,14] = (-float(3)/2*K[0]*np.ones(3))
    
    ### (N-1)th row BD
    lhs_matrix[-1,-1],lhs_matrix[-2,-2],lhs_matrix[-3,-3] = 2*K[-1]*np.ones(3)
    lhs_matrix[-1,-4],lhs_matrix[-2,-5],lhs_matrix[-3,-6] = -5*K[-1]*np.ones(3)
    lhs_matrix[-1,-7],lhs_matrix[-2,-8],lhs_matrix[-3,-9] = 4*K[-1]*np.ones(3)
    lhs_matrix[-1,-10],lhs_matrix[-2,-11],lhs_matrix[-3,-12] = -1*K[-1]*np.ones(3)
    
    ### (N-2)th row BD
    lhs_matrix[-4,-1],lhs_matrix[-5,-2],lhs_matrix[-6,-3] = (float(5)/2*K[-1]*np.ones(3)) + (2*Ks[-1]*np.ones(3))
    lhs_matrix[-4,-4],lhs_matrix[-5,-5],lhs_matrix[-6,-6] = (-9*K[-1]*np.ones(3)) + (-5*Ks[-1]*np.ones(3))
    lhs_matrix[-4,-7],lhs_matrix[-5,-8],lhs_matrix[-6,-9] = (12*K[-1]*np.ones(3)) + (4*Ks[-1]*np.ones(3))
    lhs_matrix[-4,-10],lhs_matrix[-5,-11],lhs_matrix[-6,-12] = (-7*K[-1]*np.ones(3)) + (-1*Ks[-1]*np.ones(3))
    lhs_matrix[-4,-13],lhs_matrix[-5,-14],lhs_matrix[-6,-15] = (float(3)/2*K[-1]*np.ones(3))

    return lhs_matrix

def calc_stress(sim_c,force,x_loc):
    """
    This function calculates the stress (sigma) using the following equation:
    sigma = \int^{1}_{0}{\textbf{f(s)}\textbf{x(s)} ds} with the integrand being 
    a dyadic product.
    
    Inputs: 

    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    force:      Nx3 vectorized array that contains the force acting on each
                point of the filament. 
    x_loc:      Nx3 vectorized array that contains the location of each filament
                point.
    """
    diag_index,N,s = sim_c.diag_index,sim_c.N,sim_c.s
    ### Method 1 ### (more efficient)
    dyad_prod_all = np.outer(force,x_loc)
    dyad_prod_of_int = dyad_prod_all.flatten()[diag_index].reshape(3*N,3).reshape(N,3,3)
    stress = np.trapz(y = dyad_prod_of_int,x = s,axis = 0)
    true_stress = 0.5*(stress+stress.transpose())
    return true_stress


def calc_E_elastic(sim_c,x_ss):
    """
    This function calculates the elastic energy of the filament at a given 
    point in time using the following equation: 
        E_{elastic} = \frac{1}{2}\int^{1}_{0}{|\textbf{x}_{ss}|^{2} ds}. 
    
    Inputs:
    
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    x_ss:       Nx3 vectorized array that contains the 2nd derivative of the filament
                position at each point along the filament. 
    """
    s,K = sim_c.s,sim_c.K
    integrand = K*np.linalg.norm(x_ss,axis = 1)**2
    elastic_en = 0.5*np.trapz(y = integrand,x = s)
    return elastic_en


def eval_time_euler(sim_c,filament_arc_data):
    """
    This function solves for position of the filament at the future time step using a
    general Euler method.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    t:          Current iteration of the simulation.
    """  
    
    ### Check if filament is within proximity to wall to use adaptive time step ###
    
    ### Get filament length ###
    # fil_length = rod_length(sim_c.N,sim_c.initial_loc)
    
    ### Determine fluid velocity and derivative ###
    u0,du0ds = fluid_velo(sim_c,sim_c.initial_loc)
    
    ### Calculate spatial derivative ###
    dxds, d2xds2, d3xds3, d4xds4 = spatialderiv(sim_c,sim_c.initial_loc)
    
    ### Caculate Tension and its derivative ###
    tension = solve_tension(sim_c,du0ds, dxds, d2xds2, d3xds3, d4xds4)
    dTds = first_deriv(tension,sim_c.N,sim_c.ds,1)
    
    #Obtain spatial-derivative coupled terms
    curr_params = calc_vals(sim_c,tension, dTds, dxds, d2xds2, d3xds3, d4xds4)
    Txss,Tsxs,Kxs,Kxssss,Ksxs,Ksxsss,Kssxss = curr_params    
    force = calc_force(sim_c,Txss,Tsxs,Kssxss,Ksxsss,Kxssss,'all')
    # stress = calc_stress(sim_c,force, sim_c.initial_loc)
    # elastic_energy = calc_E_elastic(sim_c,d2xds2)
     
    ### Calculate non-time and elastic velocities for steric velocity ###
    # steric_velocity_params = [u0, du0ds, dxds, d2xds2, d3xds3,
    # Txss,Tsxs,Kxs,Ksxs,Ksxsss,Kssxss]
    
    
    #Save initial data
    mu_bar,c,N = sim_c.mu_bar,sim_c.c,sim_c.N
    
    ##### Solve for next iteration #####
    
    xt = (mu_bar*u0 -(c+1)*force-(c-3)*(Kxs*np.repeat(rdot(
        dxds,d4xds4),3).reshape(N,3) + \
            2*Ksxs*np.repeat(rdot(dxds,d3xds3),3).reshape(N,3) - Tsxs))/mu_bar
                
    # xt[0,:] = (1/11)*(48*xt[2,:] - 52*xt[3,:] + 15*xt[4,:])
    # xt[1,:] = (1/11)*(28*xt[2,:] - 23*xt[3,:] + 6*xt[4,:])
    # xt[N-1,:] = (1/11)*(48*xt[N-3,:] - 52*xt[N-4,:] + 15*xt[N-5,:])
    # xt[N-2,:] = (1/11)*(28*xt[N-3,:] - 23*xt[N-4,:] + 6*xt[N-5,:])
    
    sim_c.tension = tension
    sim_c.xt = xt
    
    filament_arc_data.append_position_state(sim_c)
    return filament_arc_data
#%% Classes
class filament_parameters():
    def __init__(self,rigid_func,slender_f,mubar,zeta_f,channel_height,U_centerline,N,a,theta_start,flow_type,flip_type):
        """
        This function initializes the class and relevant filament parameters.
        """
        self.rigidity_suffix = rigid_func
        self.c = np.log(1/slender_f**2)
        self.mu_bar = mubar
        self.L = 1
        self.N = N
        self.s = np.linspace(-1/2,1/2,self.N)
        self.ds = 1/(self.N-1)
        self.theta_start = theta_start
        self.channel_upper_height = channel_height
        self.channel_lower_height = -channel_height
        self.U_centerline = U_centerline
        self.fil_end_theta = theta_start + np.pi
        self.s_arc_N = a
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
        self.zeta_use = zeta_f
        
        ##### Determine Form of filament rigidity #####
        
        if self.rigidity_suffix == 'K_constant':
            self.K = np.ones(self.s.shape[0],dtype = float)
            self.Ks = np.zeros(self.s.shape[0],dtype = float)
            self.Kss = np.zeros(self.s.shape[0],dtype = float)
        elif self.rigidity_suffix == 'K_parabola_center_l_stiff':
            self.K = 1/2 + 2*self.s**2
            self.Ks = 4*self.s
            self.Kss = 4*np.ones(self.s.shape[0],dtype = float)
        elif self.rigidity_suffix == 'K_parabola_center_m_stiff':
            self.K = 1.5 - 2*(self.s**2)
            self.Ks = -4*self.s
            self.Kss = -4*np.ones(self.s.shape[0],dtype = float)
        elif self.rigidity_suffix == 'K_linear':
            self.K = self.s+1
            self.Ks = 1*np.ones(self.s.shape[0],dtype = float)
            self.Kss = 0*np.ones(self.s.shape[0],dtype = float)
        elif self.rigidity_suffix == 'K_dirac_center_l_stiff':
            self.K = 1-0.5*np.exp(-100*self.s**2)
            self.Ks = 100*self.s*np.exp(-100*self.s**2)
            self.Kss = np.exp(-100*self.s**2)*(100-2e4*self.s**2)
        elif self.rigidity_suffix == 'K_dirac_center_l_stiff2':
            self.K = 1-0.5*np.exp(-500*self.s**2)
            self.Ks = 500*self.s*np.exp(-500*self.s**2)
            self.Kss = np.exp(-500*self.s**2)*(500-5e5*self.s**2)
        elif self.rigidity_suffix == 'K_dirac_center_m_stiff':
            self.K = 1+np.exp(-100*self.s**2)
            self.Ks = -200*self.s*np.exp(-100*self.s**2)
            self.Kss = 200*np.exp(-100*self.s**2)*(200*self.s**2-1)
        elif self.rigidity_suffix == 'K_parabola_shifted':
            self.K = 1.5-0.5*(self.s-0.5)**2
            self.Ks = -1*self.s-0.5
            self.Kss = -1*np.ones(self.s.shape[0],dtype = float)
        elif self.rigidity_suffix == 'K_error_function':
            self.K = special.erf(10*self.s)+2
            self.Ks = (20/np.sqrt(np.pi))*np.exp(-100*self.s**2)
            self.Kss = (-4000*self.s/np.sqrt(np.pi))*np.exp(-100*self.s**2)  
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff':
            self.K = 1-0.5*np.exp(-100*(self.s+0.25)**2)
            self.Ks = 100*(self.s+0.25)*np.exp(-100*(self.s+0.25)**2)
            self.Kss = np.exp(-100*(self.s+0.25)**2)*-2e4*(self.s**2+0.5*self.s+0.0575)
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff2':
            self.K = 1-0.5*np.exp(-500*(self.s+0.25)**2)
            self.Ks = 500*(self.s+0.25)*np.exp(-500*(self.s+0.25)**2)
            self.Kss = np.exp(-500*(self.s+0.25)**2)*-5e5*(self.s**2+0.5*self.s+0.0615)
        elif self.rigidity_suffix == 'K_dirac_left_l_stiff3':
            self.K = 1-0.5*np.exp(-1000*(self.s+0.25)**2)
            self.Ks = 1000*(self.s+0.25)*np.exp(-1000*(self.s+0.25)**2)
            self.Kss = np.exp(-1000*(self.s+0.25)**2)*-2e6*(self.s**2+0.5*self.s+0.062)
            
        ##### Initialize parameters for fast dyadic calculations #####
        
        self.matrix_number = np.arange(0,(3*self.N)**2, dtype=np.int64()).reshape(3*self.N,3*self.N)
        self.indices = np.repeat(np.arange(0,3*self.N).reshape((self.N,3)), 3, axis=0)
        self.diag_index = np.take_along_axis(self.matrix_number, self.indices, axis=1).flatten()
        
        
    def configure_coord(self):
        """
        This function intializes and calculates the position coordinates of the 
        filament. It first calculates the coordinates of the arc and followed 
        by the arm. 
        """
        if self.flow_type == 'Poiseuille':
            if self.flip_type == 'Up':
                
                ### Calculate coordinates of arc ###
                self.xarc =  1/np.pi*(self.arc_radius)*-np.cos(self.s_arc_loc)
                self.yarc = 1/np.pi*(self.arc_radius)*np.sin(self.s_arc_loc)
                self.zarc = np.zeros(self.s_arc_loc.shape[0],dtype = float)
                self.arcloc = np.column_stack((self.xarc,self.yarc,self.zarc))
                
                ### Calculate coordinates of arm ###
                self.dxds_arc = first_deriv(base_array = self.arcloc,
                                            N = self.arcloc.shape[0],
                                            ds = self.ds,dim = 2)
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
                self.initial_loc = np.concatenate((self.armloc,self.arcloc),axis = 0)
                self.initial_loc += np.column_stack((0.2*np.ones(self.N,dtype = float),
                                                                 0.15*np.ones(self.N,dtype = float),
                                                                 np.zeros(self.N,dtype = float)))
                
                
                
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
                self.dxds_arc = first_deriv(base_array = self.arcloc,
                                            N = self.arcloc.shape[0],
                                            ds = self.ds,dim = 2)
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
                self.initial_loc = np.concatenate((self.armloc,self.arcloc),axis = 0)
                self.initial_loc += np.column_stack((-0.20*np.ones(self.N,dtype = float),
                                                                 0.30*np.ones(self.N,dtype = float),
                                                                 np.zeros(self.N,dtype = float)))
                
        if self.arc_end_angle < 0:
            self.arc_end_angle += np.pi
                
    def adjust_arc_end_coord(self):
        """
        This function adjusts the end coordinate of the arc component of the filament
        so that the end coordinate is technically force-free and torque-free. In order 
        to accomplish this, the end coordinate is on the same tangent line as the 
        point on the filament that precedes this one. 
        """
        self.dot_theta = np.arccos(np.dot(self.initial_loc[-1,:],self.initial_loc[-2,:])/\
                                   (np.linalg.norm(self.initial_loc[-1,:]) * np.linalg.norm(self.initial_loc[-2,:])))
            
        self.new_end_loc = (self.initial_loc[-1,:]/np.cos(self.dot_theta)).reshape(1,3)
        self.initial_loc[-1,:] = self.new_end_loc
        

class end_parameters():
    def __init__(self):
        self.position_stats_lst = []
        
    def append_position_state(self,fil_params):
        self.position_state_params = {"N": fil_params.N,
                                      "s": fil_params.s,
                                      's Start': fil_params.s[0],
                                      's End': fil_params.s[-1],
                                      "Mu_bar": fil_params.mu_bar,
                                      "Arc Start Angle":np.degrees(fil_params.theta_start),
                                      "Arc End Angle": np.degrees(fil_params.fil_end_theta),
                                      "Arm Angle": np.degrees(fil_params.arc_end_angle),
                                      "Arc Parameterized Length": fil_params.s_arc_N,
                                      "s Start for Arc":fil_params.s[-fil_params.s_arc_N],
                                      "s End for Arc":fil_params.s[-1],
                                      "Straight Arm Parameterized Length": fil_params.N - fil_params.s_arc_N,
                                      "Average Filament Velocity-x": fil_params.xt[:,0].mean(),
                                      "Average Filament Velocity-y": fil_params.xt[:,1].mean(),
                                      "Average Filament Velocity-z": fil_params.xt[:,2].mean(),
                                      "Average Filament Arm Velocity-x": fil_params.xt[:-fil_params.s_arc_N,0].mean(),
                                      "Average Filament Arm Velocity-y": fil_params.xt[:-fil_params.s_arc_N,1].mean(),
                                      "Average Filament Arm Velocity-z": fil_params.xt[:-fil_params.s_arc_N,2].mean(),
                                      "Average Filament Arc Velocity-x": fil_params.xt[-fil_params.s_arc_N:,0].mean(),
                                      "Average Filament Arc Velocity-y": fil_params.xt[-fil_params.s_arc_N:,1].mean(),
                                      "Average Filament Arc Velocity-z": fil_params.xt[-fil_params.s_arc_N:,2].mean(),
                                      "Average Filament COM-x":fil_params.xt[:,0].mean(),
                                      "Average Filament COM-y":fil_params.xt[:,1].mean(),
                                      "Average Filament COM-z":fil_params.xt[:,2].mean(),
                                      "Filament Position-x": fil_params.initial_loc[:,0],
                                      "Filament Position-y": fil_params.initial_loc[:,1],
                                      "Filament Position-z": fil_params.initial_loc[:,2],
                                      "Filament Tension": fil_params.tension}
        self.position_stats_lst.append(self.position_state_params)
    
    
    def compile_position_states(self):
        """
        This function converts the dictionary of all positional data into a Pandas
        DataFrame.
        """
        self.all_position_stats_data_df = pd.concat(
            [pd.DataFrame.from_dict(i) for i in self.position_stats_lst],ignore_index = True)



def run_simulation(filament_arc_data,rigid_type,mu_bar,arc_theta,flip_type):
    sim_c = filament_parameters(rigid_func = rigid_type,
                                slender_f = 0.01,mubar = mu_bar,zeta_f = 125*mu_bar,
                                channel_height = 0.5,U_centerline = 1,
                                N = 101,a = 30,theta_start = arc_theta,
                                flow_type = 'Poiseuille',flip_type = flip_type)
    sim_c.configure_coord()
    sim_c.adjust_arc_end_coord()
    
    ### Run 1 Euler Step on problem ###
    filament_arc_data = eval_time_euler(sim_c,filament_arc_data)
    
    return filament_arc_data
#%% Run Scripts
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_directory",
                        help="Specify the parent directory of the Migration Data",
                    type = str)
    parser.add_argument("rigidity_type",
                        help = "Specify what kind of rigidity profile this simulation will run on",
                        type = str,
                        choices = {"K_constant","K_parabola_center_l_stiff",'K_parabola_center_l_stiff',
                                    'K_parabola_center_m_stiff','K_linear','K_dirac_center_l_stiff',
                                    'K_dirac_center_l_stiff2','K_dirac_center_m_stiff','K_parabola_shifted',
                                    'K_error_function','K_dirac_left_l_stiff','K_dirac_left_l_stiff2',
                                    'K_dirac_left_l_stiff3'})
    parser.add_argument("mu_bar",
                        help = "Specify what the mu_bar value will be to run the simulations on",
                        type = int)
    # args = parser.parse_args()
    ## Comment this line below when running it as through a separate terminal
    args = parser.parse_args(['C://Users//super//OneDrive - University of California, Davis//School//UCD_Files//Work//00_Projects//02_Shear_Migration//00_Scripts//01d_J_Shape_Calculations//Plots/',
                              'K_constant','100000'])
    
    filament_arc_data_up = end_parameters()
    filament_arc_data_down = end_parameters()
    
    theta_test_up = 115*np.pi/200
    filament_arc_data_up = run_simulation(filament_arc_data_up,args.rigidity_type,args.mu_bar,theta_test_up,flip_type = 'Up')
    filament_arc_data_up.compile_position_states()
    
    theta_test_down = 285*np.pi/200
    filament_arc_data_down = run_simulation(filament_arc_data_down,args.rigidity_type,args.mu_bar,theta_test_down,flip_type = 'Down')
    filament_arc_data_down.compile_position_states()
    
#%%

fig,axes = plt.subplots(ncols = 2,figsize = (10,7))
axes[0].plot(filament_arc_data_up.all_position_stats_data_df['Filament Position-x'],
             filament_arc_data_up.all_position_stats_data_df['Filament Position-y'],
             color = 'red',label = 'Upward Flip')
axes[0].plot(filament_arc_data_down.all_position_stats_data_df['Filament Position-x'],
             filament_arc_data_down.all_position_stats_data_df['Filament Position-y'],
             color = 'blue',label = 'Downward Flip')

axes[1].plot(filament_arc_data_up.all_position_stats_data_df['s'],
          filament_arc_data_up.all_position_stats_data_df['Filament Tension'],color = 'red',
          label = 'Upward Tension')
axes[1].plot(filament_arc_data_down.all_position_stats_data_df['s'],
          filament_arc_data_down.all_position_stats_data_df['Filament Tension'],color = 'blue',
          label = 'Downward Tension')
axes[0].plot(np.array(filament_arc_data_up.all_position_stats_data_df['Filament Position-x'])[72],
             np.array(filament_arc_data_up.all_position_stats_data_df['Filament Position-y'])[72],
             color = 'cyan',marker = 'o',markersize = 4)
axes[0].plot(np.array(filament_arc_data_down.all_position_stats_data_df['Filament Position-x'])[72],
             np.array(filament_arc_data_down.all_position_stats_data_df['Filament Position-y'])[72],
             color = 'cyan',marker = 'o',markersize = 4)
axes[0].set_xlim(-0.6,0.6)
axes[0].set_ylim(-0.6,0.6)
axes[0].set_xticks(np.linspace(-0.5,0.5,5))
axes[0].set_yticks(np.linspace(-0.5,0.5,5))
axes[1].set_xticks(np.linspace(-0.5,0.5,5))

axes[0].set_xlabel(r"$x$",fontsize = 12,labelpad = 5)
axes[0].set_ylabel(r"$y$",fontsize = 12,labelpad = 5)
axes[1].set_xlabel(r"$s$",fontsize = 12,labelpad = 5)
axes[1].set_ylabel(r"$T(s)$",fontsize = 12,labelpad = 5)

for ax in axes:
    ax.set_aspect((ax.get_xlim()[1] - ax.get_xlim()[0])/(ax.get_ylim()[1] - ax.get_ylim()[0]))
    ax.legend(loc = 'lower left')
    
plt.show()

print({"Upward Flip Filament Arm Angle":filament_arc_data_up.all_position_stats_data_df['Arm Angle'].unique()[0],
       "Downward Flip Filament Arm Angle":filament_arc_data_down.all_position_stats_data_df['Arm Angle'].unique()[0]})

# print({"Upward Flip Filament Arm Average Velocity-y": \
#        filament_arc_data_up.all_position_stats_data_df['Average Filament Arm Velocity-y'].unique()[0],
#        "Downwward Flip Filament Arm Average Velocity-y": \
#            filament_arc_data_down.all_position_stats_data_df['Average Filament Arm Velocity-y'].unique()[0]})

# print({"Upward Flip Filament Arm Average Velocity-x": \
#        filament_arc_data_up.all_position_stats_data_df['Average Filament Arm Velocity-x'].unique()[0],
#        "Downwward Flip Filament Arm Average Velocity-x": \
#            filament_arc_data_down.all_position_stats_data_df['Average Filament Arm Velocity-x'].unique()[0]})

print({"Upward Flip Filament Average Velocity-y": \
       filament_arc_data_up.all_position_stats_data_df['Average Filament Velocity-y'].unique()[0],
       "Downwward Flip Filament Average Velocity-y": \
           filament_arc_data_down.all_position_stats_data_df['Average Filament Velocity-y'].unique()[0]})    

print({"Upward Flip Max Tension": \
       filament_arc_data_up.all_position_stats_data_df['Filament Tension'].max(),
       "Downwward Flip Max Tension": \
           filament_arc_data_down.all_position_stats_data_df['Filament Tension'].max()})
    
print({"Upward Flip Min Tension": \
       filament_arc_data_up.all_position_stats_data_df['Filament Tension'].min(),
       "Downwward Flip Min Tension": \
           filament_arc_data_down.all_position_stats_data_df['Filament Tension'].min()})

