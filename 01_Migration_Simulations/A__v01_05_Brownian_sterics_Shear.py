# -*- coding: utf-8 -*-
"""
FILE NAME:      A__v01_04_Brownian_sterics_Poiseuille.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    Given an analytical form for the rigidity of the filament and 
                a certain fluid velocity profile, this script will calculate 
                and predict the movment and tension of the filament for a 
                specified duration. This script also corrects the filament velocity
                due to the sterics associated with 2 parallel walls.

INPUT
FILES(S):       N/A

OUTPUT
FILES(S):       1) .NPY file that contains all positions of each discretized
                position of the filament for the duration of the simulation.
                2) .NPY file that contains all tension values of each discretized
                position of the filament for the duration of the simulation.
                3) .NPY file that contains the numerical values of the rigidity
                of the function. 
                4) .NPY file that contains the length of the filament at each
                timestep. 
                5) .NPY file that contains the stresses of the filament at each 
                timestep. 
                6) .NPY file that contains the elastic energy of the filament 
                at each timestep.
                7) .CSV file that lists all parameters used for run. 


INPUT
ARGUMENT(S):    1) Main Output directory: The directory that will house all of the
                output files associated with the simulation; if it doesn't exist,
                it will be created.
                2) Rigidity Type: The specific rigidity profile that will be 
                used for the simulation:
                    K_constant:                     K(s) = 1
                    K_parabola_center_l_stiff:      K(s) = /frac{1}{2} + 2s^{2}
                    K_parabola_center_m_stiff:      K(s) = /frac{3}{2} - 2s^{2}
                    K_linear:                       K(s) = s + 1
                    K_dirac_center_l_stiff:         K(s) = 1-\frac{1}{2}\exp^{-100s^{2}}
                    K_dirac_center_l_stiff2:        K(s) = 1-\frac{1}{2}\exp^{-500s^{2}}
                    K_dirac_center_m_stiff:         K(s) = 1+\exp^{-100s^{2}}
                    K_parabola_shifted:             K(s) = \frac{3}{2} - \frac{1}{2}\left(s-\frac{1}{2}\right)^{2}
                    K_error_function:               K(s) = 2+\erf(10s)
                    K_dirac_left_l_stiff:           K(s) = 1-\frac{1}{2}\exp^{-100\left(s+frac{1}{4}\right)^{2}}
                    K_dirac_left_l_stiff2:          K(s) = 1-\frac{1}{2}\exp^{-500\left(s+frac{1}{4}\right)^{2}}
                    K_dirac_left_l_stiff3:          K(s) = 1-\frac{1}{2}\exp^{-1000\left(s+frac{1}{4}\right)^{2}}
                3) Mu_bar Value: The mu_bar value that you want to run your simulations at. 
                4) Replicate Number:  The replicate number 
                (for output directory nomenclature purposes)


CREATED:        1Jun20

MODIFICATIONS
LOG:
15Jun20         1) Added compatibility for non-uniform rigidity of the filament.
15Jun20         2) Vectorized derivative calculations for faster computation
                time.
01Aug20         3) Changed calculation of tension from TDMA solver function
                to np.linalg.solve.
10Nov20         4) Added functionality for semi-implicit method (4th order 
                spatial derivative term only). Re-arranged calculations of 
                parameters to account for this. 
12Nov20         5) Removed functionality to plot .MP4 files for filament 
                movmeent and tension. Functionality has been moved to 
                Plot_Results.py.
03Apr21         6) Cleaned up code. Added global variables to each function.
03Apr21         7) Added function to calculate tensile and rigidity forces.
20Jul21         8) Added functions to calculate various spatial derivatives to 
                shorten code.
20Jul21         9) Added functions to calculate stresses and elastic energy. 
16Aug21         10) Adjusted time scaling for numpy array outputs to conserve memory.
01Sep21         11) Added boundary conditions for torque-free and force-free filament
                at end of Euler step. Adjusted saving steps with small Euler-step.
26Sep21         12) Created a class to house all parameters and data. Simulation
                is now initiated if "if __name__ == __main__".
28Sep21         12) Allowed filament to reverse direction of rotation based on 
                angle and deflection criterion.
29Sep21         13) Code now reformatted for Brownian motion.
17Jan22         14) Added functionality for argparse (input arguments before 
                running code). Added functionality for logging basic 
                checkpoint messages. Changed input arguments of several functions. 
20May22         15) Fixed force-free boundary conditions in the semi-implicit steps.
02Jun22         16) Adopted corrective velocity due to sterics associated with 2 parallel walls.
27Jun22         17) Implemented adaptive time stepping. Corrected Brownian forces during adaptive steps.
12Jul22         18) Adaptive time stepping will now occur to align at the time iterations when the data is saved.
                Data is now saved at uniform time points.
18Jul22         19) Script now runs for Poiseuille flow.
06Sep22         20) Script can now restart when it blows up due to instability.
14Sep22         21) Additional error logging and parameter saving when script blows up due to instability.
06Feb23         22) If code fails, option to restart the entire code. 

    
            
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.8.8

VERSION:        1.5

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:    1) Non-local operator Kappa. 

NOTE(S):        N/A

"""


import sys, os, math, time, random, argparse, logging, warnings, re, glob
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from fractions import Fraction
from scipy import special, sparse

#suppress warnings
warnings.filterwarnings('ignore') #Suprress overflow warning if vector blows up

#Import other scripts
from create_dir import create_dir
#%% ####################### Functions #######################


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
    k_flow_phase,k_flow_factor = sim_c.kflow_phase,sim_c.kflow_freq
    
    u0 = np.zeros([N,3],dtype = float)
    
    ### Shear flow ###
    u0[:,0] = x_vec[:,1]
    
    ### Poiseuille Flow for H = L ###
    # u0[:,0] = U_centerline*(1-(x_vec[:,1]**2/H**2))
    ### Extensional Flow ###
    # Find true center position (average of x, y, z)
   
    # u0[:,0] = -x_vec[:,0] - x_vec[np.where(s == 0)[0][0],0]   # Change to Relative to center of x-component
    # u0[:,1] = x_vec[:,1]  - x_vec[np.where(s == 0)[0][0],1]   # Change to relative to center of y-component
    
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
    

def calc_Brownian(sim_c,dxds):
    """
    This function calculates the Brownian forces for the filament. Brownian 
    forces are derived from a Gaussian distribution with a mean of 0 and 
    variance of 1.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    dxds:       Nx3 vectorized array that represents the first spatial derivative
                of the filament position. 
    """
    N,ds,c = sim_c.N, sim_c.ds,sim_c.c
    diag_index,rng = sim_c.diag_index,sim_c.rng
    dt = sim_c.true_dt
    
    brownian_distr = np.column_stack((rng.normal(0,1,(N)),rng.normal(0,1,(N)),np.zeros(N,dtype = float)
                                  ))
    pref_1 = np.sqrt(1/(c+1))
    pref_2 = -1-np.sqrt((c+1)/(2*(c-1)))
    #Method 1
    dyad_component = (np.identity(3*N) + (pref_2*np.outer(dxds,dxds)))
    resist_matr = pref_1*dyad_component
    resist_matr = resist_matr.flatten()[diag_index].reshape(3*N,3)
    dot_prod = np.multiply(resist_matr,np.repeat(brownian_distr,3,axis = 0))
    dot_prod = np.sum(dot_prod,axis = 1).reshape(N,3)
    brownian_f = np.sqrt(2/(ds*dt))*dot_prod
    
    return brownian_f


def calculate_N_ex(sim_c,params):
    """
    This function calculates the non-semi implicit terms.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    params:     list of arrays that contain components needed to solve explicit terms
                in the semi-implicit time-stepping. 
    """
    N,c, mu_bar,llp = sim_c.N,sim_c.c, sim_c.mu_bar, sim_c.llp
    
    #Unpack params
    u0, du0ds, dxds, d2xds2, d3xds3,\
    Txss,Tsxs,Kxs,Ksxs,Ksxsss,Kssxss,brownian = params
    
    N_ex = (mu_bar*u0) - ((c+1)*(-Tsxs + Kssxss - Txss + 2*Ksxsss + llp*brownian) + 
                          (c-3)*(-Tsxs + 2*Ksxs*np.repeat(rdot(dxds,d3xds3),3).reshape(N,3) + \
                                 llp*dxds*np.repeat(rdot(dxds,brownian),3).reshape(N,3)))                                    
    return N_ex

def solve_tension(sim_c,du0ds,dxds,d2xds2,d3xds3,d4xds4,brownian):
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
    llp = sim_c.llp
    
    dxids = first_deriv(brownian,N,ds, 2)
    
    
    # Evaluating Tension Equation with BC: Tension = 0 at ends of Filament 
    a = np.ones(N-3)*(-2*(c-1)/ds**2) # Lower and Upper Diag
    b = np.ones(N-2)*(4*(c-1)/ds**2)+ (c+1)*rdot(d2xds2,d2xds2)[1:N-1] # Center Diag  
    d = ((mu_bar*rdot(du0ds,dxds))+ ((5*c-3)*(Kss*rdot(d2xds2,d2xds2))) + \
          (4*(4*c-3)*(Ks*rdot(d2xds2,d3xds3))) + \
            ((7*c-5)*K*rdot(d2xds2,d4xds4))+(6*(c-1)*K*rdot(d3xds3,d3xds3)) -\
                2*(c-1)*llp*rdot(dxds,dxids) - (c-3)*llp*rdot(brownian,d2xds2) - \
          sim_c.zeta_use*(1-rdot(dxds,dxds)))[1:N-1] # RHS-non constant K
        
    ### Evluate tension ###
    A = sparse.diags([a,b,a],offsets = [-1,0,1],shape = (N-2,N-2)).toarray()
    tension = np.insert(np.linalg.solve(A,d),(0,N-2),0)
    
    return tension


def calc_force(sim_c,Txss,Tsxs,Kssxss,Ksxsss,Kxssss,brownian,f_type):
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
    llp = sim_c.llp
    
    if f_type == 'rigid':
        force = Kssxss + 2*Ksxsss + Kxssss + llp*brownian
    elif f_type == 'all':
        force = -Tsxs - Txss + Kssxss + 2*Ksxsss + Kxssss + llp*brownian
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


def calc_angle(sim_c,x_vec):
    """
    This function calculates the orientation of the filament by calculating 
    the average angle at each point across the filament and averaging them.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    x_vec:      Nx3 array that contains the coordinates at each point 
                along the filament. 
    """
    s,centerline_idx = sim_c.s,sim_c.centerline_idx
    #Center filament at origin
    adj_fil_loc = x_vec[:,:] - x_vec[centerline_idx,:]
    #Adjust for arclength
    adj_loc = np.divide(adj_fil_loc[:,:],np.column_stack((s,s,s)),
                                out = np.zeros_like(x_vec[:,:]),
                                where=np.column_stack((s,s,s))!=0)
    #Calculate angle
    angle_adj_loc = np.arctan(np.divide(adj_loc[:,1],adj_loc[:,0],out = np.zeros_like(adj_loc[:,1]),where=adj_loc[:,0]!=0))
    angle_adj_loc[angle_adj_loc<0] = angle_adj_loc[angle_adj_loc<0] + np.pi #Adjust for negative angles
    fil_angle = np.average(angle_adj_loc)
    return fil_angle


def calc_deflect(sim_c,x_vec,angle):
    """
    This function calculates the deflection of the filament by measuring the 
    different between the y-coordinates of the filament to a "Base state" of the same
    angle.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    x_vec:      Nx3 array that contains the coordinates at each point 
                along the filament. 
    angle:      average angle of the filament.
    """
    
    s,centerline_idx = sim_c.s,sim_c.centerline_idx
    
    base_state_fil = np.column_stack((s*np.cos(angle),s*np.sin(angle),s*0))
    fil_deflect = x_vec[:,:] - x_vec[centerline_idx,:] #Second term adjusts for any translation of filament
    fil_deflect_all = (fil_deflect[:,1] - base_state_fil[:,1])**2 
    fil_deflect = np.sqrt(np.sum(fil_deflect_all))
    return fil_deflect


def det_rotation(sim_c,fil_loc):
    """
    This function calculates the angle of each position along the filament and
    calculates the average angle. If the average angle is roughly 1*pi/9 while
    the shear flow moves in the postive x-direction or 8*pi/9 while the shear flow
    moves in the negative x-direction, it will reverse the direction of the shear
    flow.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    fil_loc:    Nx3 vectorized array that contains the coordinates of 
                each filament position at the curren time.
    
    """
    
    filament_angle = calc_angle(sim_c,fil_loc)
    filament_deflect = calc_deflect(sim_c,fil_loc, filament_angle)
        
    
    return sim_c,filament_angle, filament_deflect

def steric_velocity(sim_c,x_loc,non_elastic_U):
    """
    This function applies the contact algorithm to reverse the direction of the
    velocity and allow the filament to glide along the boundary walls should it 
    come into close contact with them:
        
    Inputs:
    
    sim_c:              Class that contains all parameters needed for the simulation 
                        and arrays to store data.
    x_loc:              Nx3 vectorized array that containss the location of each filament
                        point.
    non_elastic_U:      Nx3 array that contains the summation of all the non-elastic
                        and non-time velocity terms.
    """
    low_wall_prox = np.where(np.abs(x_loc[:,1] - sim_c.channel_lower_height) <=sim_c.gap_criteria)[0]
    high_wall_prox = np.where(np.abs(x_loc[:,1] - sim_c.channel_upper_height) <=sim_c.gap_criteria)[0]
    new_U = non_elastic_U.copy()
    if low_wall_prox.any() and not high_wall_prox.any(): #Check if filament is within distance of lower wall and reverse direction of velocity
        new_U = calculate_lower_wall_velocity(sim_c,low_wall_prox,x_loc,new_U)
    elif not low_wall_prox.any() and high_wall_prox.any(): #Check if filament is within distance of higher wall and reverse direction of velocity 
        new_U = calculate_higher_wall_velocity(sim_c,high_wall_prox,x_loc,new_U)
    elif low_wall_prox.any() and high_wall_prox.any(): #Check if filament is within distance of both walls and reverse direction of velocity 
        new_U = calculate_lower_wall_velocity(sim_c,low_wall_prox,x_loc,new_U)
        new_U = calculate_higher_wall_velocity(sim_c,high_wall_prox,x_loc,new_U)
    adj_U = np.column_stack((non_elastic_U[:,0],new_U[:,1],non_elastic_U[:,2]))
    return adj_U


def calculate_lower_wall_velocity(sim_c,low_wall_points,x_loc,non_elastic_U):
    """
    This function reverses the direction of the filament velocity near the lower wall.
    Near the lower wall, the filament should have a positive velocity.
        
    Inputs:
    
    sim_c:              Class that contains all parameters needed for the simulation 
                        and arrays to store data.
    low_wall_points:    Indices on s that indicate which points on the filament are near the low wall.
    x_loc:              Nx3 vectorized array that containss the location of each filament
                        point.
    non_elastic_U:      Nx3 array that contains the summation of all the non-elastic
                        and non-time velocity terms.
    """
    dist_to_low_wall =x_loc[low_wall_points,1] - sim_c.channel_lower_height
    adj_U_lower_normal = np.multiply((1-(sim_c.gap_criteria/dist_to_low_wall)**sim_c.steric_velo_exp),non_elastic_U[low_wall_points,1])
    correct_U_lower = np.array([non_elastic_U[low_wall_points,1],adj_U_lower_normal]).max(axis = 0) #Velocity should be in positive direction
    non_elastic_U[low_wall_points,1] = correct_U_lower
    return non_elastic_U


def calculate_higher_wall_velocity(sim_c,high_wall_points,x_loc,non_elastic_U):
    """
    This function reverses the direction of the filament velocity near the lower wall.
    Near the lower wall, the filament should have a positive velocity.
        
    Inputs:
    
    sim_c:              Class that contains all parameters needed for the simulation 
                        and arrays to store data.
    low_wall_points:    Indices on s that indicate which points on the filament are near the low wall.
    x_loc:              Nx3 vectorized array that containss the location of each filament
                        point.
    non_elastic_U:      Nx3 array that contains the summation of all the non-elastic
                        and non-time velocity terms.
    """
    dist_to_high_wall =x_loc[high_wall_points,1] - sim_c.channel_upper_height
    adj_U_lower_normal = np.multiply((1-(sim_c.gap_criteria/dist_to_high_wall)**sim_c.steric_velo_exp),non_elastic_U[high_wall_points,1])
    correct_U_higher = np.array([non_elastic_U[high_wall_points,1],adj_U_lower_normal]).min(axis = 0) #Velocity should be in negative direction
    non_elastic_U[high_wall_points,1] = correct_U_higher
    return non_elastic_U
    
    
class Constants:
    """
    This class will bundle up all constants and parameters needed in the simulations to be 
    easily accessed in the functions due to multi-processing implementation.
    """
    
    ########
    
    retry_type = True
    
    def __init__(self,end_dir,rigid_func,slender_f,mubar,true_zeta_f,adpt_zeta_f,top, bottom,
                 U_centerline,k_flow_phase,k_flow_factor,gap_criteria,force_power,
    channel_height,vert_displace,N,lp,reg_dt,adpt_dt,end_time,save_time):
        """
        Initialize the class
        """

        ##### Traditional Constants #####
        self.output_dir = end_dir
        self.rigidity_suffix = rigid_func
        self.c = np.log(1/slender_f**2)
        self.mu_bar = mubar
        self.true_zeta = true_zeta_f
        self.adpt_zeta = adpt_zeta_f
        self.N = N
        self.numerator = top
        self.denominator = bottom
        self.theta = top*np.pi/bottom
        self.L = 1
        self.lp = lp
        self.llp = np.sqrt(self.L/self.lp)
        self.s = np.linspace(-(self.L/2),(self.L/2),N)
        self.centerline_idx = np.where(self.s == 0)[0][0]        
        self.ds = 1/(self.N-1)
        self.it_count = 0
        self.record_it_count = 0
        self.end_time = end_time
        self.curr_time = 0
        self.U_centerline = U_centerline
        self.kflow_phase = k_flow_phase
        self.kflow_freq = k_flow_factor
        
        #Parameters due to normal time stepping
        self.true_dt = reg_dt
        self.true_r1 = self.true_dt/(self.ds**4)
        self.true_r2 = self.true_dt
        
        #Parameters due to adaptive time setting
        self.adpt_dt = adpt_dt
        self.adpt_r1 = self.adpt_dt/(self.ds**4)
        self.adpt_r2 = self.adpt_dt
        self.save_time = save_time
        
        
        # Pre-calculate array length based on total time and when to save data #
        self.ar_length = np.linspace(0,self.end_time,int(self.end_time/self.save_time)+1).size
        self.ar_idx = 0

        #Parameters to use when calculating Brownian motion, tension#
        self.r1_use = 0
        self.r2_use = 0
        self.dt_use = 0
        self.zeta_use = 0
        
        # Checkpoints if filament is nearby boundaries of system #
        self.adpt_dt_use = False
        self.adpt_dt_Brownian_counter = 0
        self.adpt_dt_Brownian_ratio = int(self.true_dt/self.adpt_dt)
        
        # Random seed generator #
        self.rand_seed = random.randrange(0,1e6) + os.getpid() + int(time.time()/1e6)
        self.rng = np.random.default_rng(self.rand_seed)
        
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
        
        #### Initialize starting filament location #####
        self.vertical_displacement = vert_displace
        self.start_x = self.s*np.cos(self.theta)
        self.start_y = self.s*np.sin(self.theta) +self.vertical_displacement
        self.start_z = np.zeros(self.N)
        self.initial_loc = np.column_stack((self.start_x,self.start_y,self.start_z))
        
        #### Initialize parameters for confined geometry ####
        
        self.channel_upper_height = channel_height
        self.channel_lower_height = -channel_height
        self.steric_velo_exp = force_power
        self.gap_criteria = gap_criteria*self.L
                
        ##### Initialize Arrays for storing numerical data #####
        
        self.allstate = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_tension_states = np.zeros((self.N,self.ar_length),dtype = float) 
        self.all_dTds_states = np.zeros((self.N,self.ar_length),dtype = float) 
        self.all_u0_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.expl_U_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_du0ds_states = np.zeros((self.N,3,self.ar_length),dtype = float) 
        self.all_forces_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_brownian_f_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_stress_states = np.zeros((3,3,self.ar_length),dtype = float)
        self.all_elastic_states = np.zeros(self.ar_length,dtype = float)
        
        ##### Keep track of all spatial derivatives #####
        self.all_dxds_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_d2xds2_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_d3xds3_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_d4xds4_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        
        ##### Keep track of all terms coupled to T & K #####
        self.all_Tsxs_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_Txss_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_Kssxss_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_Ksxsss_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_Kxssss_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_Kxs_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        self.all_Ksxs_states = np.zeros((self.N,3,self.ar_length),dtype = float)
        
        self.length = np.zeros(self.ar_length,dtype = float) #Track filament length
        self.angle_calculations = np.zeros(self.ar_length,dtype = float)
        self.deflection_calculations = np.zeros(self.ar_length,dtype = float)
        
        ##### Keep track whether each datapoint is using normal or adaptive time step #####
        self.all_time_vals = np.round(np.linspace(0,self.end_time,int(self.end_time/self.save_time)+1),int(-np.log10(self.save_time)))
        
        self.all_data = [self.allstate,self.expl_U_states,self.all_u0_states,self.all_du0ds_states,self.all_dxds_states,self.all_d2xds2_states, #0-5
                    self.all_d3xds3_states, self.all_d4xds4_states, self.all_Txss_states, self.all_Tsxs_states, self.all_Kxs_states, #6-10
                    self.all_Kxssss_states, self.all_Ksxs_states, self.all_Ksxsss_states, self.all_Kssxss_states,self.all_brownian_f_states, #11-15
                    self.all_forces_states, self.all_stress_states, #16-17
                    self.all_tension_states, self.all_dTds_states, #18-19
                    self.all_elastic_states, self.length,self.angle_calculations,self.deflection_calculations] #20-23
            
    ##### Instances #####
            
    def add_true_time(self):
        """
        This method keeps track of the current time based on the normal time stepping. It also rounds the current time 
        to keep precision.
        """
        self.curr_time += self.true_dt
        self.curr_time = np.round(self.curr_time,int(-np.log10(self.adpt_dt)))
    
    def add_adpt_time(self):
        """
        This method keeps track of the current itme based on the adaptive time stepping. It also rounds the current
        time to keep precision.
        """
        self.curr_time += self.adpt_dt
        self.curr_time = np.round(self.curr_time,int(-np.log10(self.adpt_dt)))
        
    def add_ar_idx(self):
        """
        This method adds 1 to the current array index whenever the simulation time is multiple to whenever the 
        time step for recording data.
        """
        self.ar_idx += 1
        
    def det_proximity_wall(self,x_vec):
        """
        This instance checks to see if the filament is near the top or bottom wall at all.
        """
        self.check_high_wall_count = np.where(np.abs(x_vec[:,1] - self.channel_upper_height) <= 1.2*self.gap_criteria)[0].size
        self.check_low_wall_count = np.where(np.abs(x_vec[:,1] - self.channel_lower_height) <= 1.2*self.gap_criteria)[0].size
        if self.check_high_wall_count or self.check_low_wall_count: #If filament is near the top or bottom wall
            #These 2 lines of code should get triggered if:
            # 1) Current time has to be a multiple of save time iteration AND
            # 2) previous iteration was a true time step
            
            if not self.adpt_dt_use and (np.round(self.curr_time,int(-np.log10(self.adpt_dt))) == self.all_time_vals[self.ar_idx + 1]): 
                self.wall_proximity_near()
                self.update_Brownian_counter()
                logging.info("Simulation is now entering adaptive time stepping at t = {:.3e}".format(self.curr_time))
            #This should get triggered for instances of adaptive time stepping
            elif self.adpt_dt_use:
                #These lines of code should get triggered if 
                self.wall_proximity_near()
                self.update_Brownian_counter()
            #Act if filament is further away from the wall-Will need to wait until current time is a multiple of save time iteration
            else:
                self.wall_proximity_far()
                self.reset_Brownian_counter()
        else: #If filament is away from any of the walls
            #Keep running adaptive time step until current time is multiple of save time iteration
            if self.adpt_dt_use and not (np.round(self.curr_time,int(-np.log10(self.adpt_dt))) == self.all_time_vals[self.ar_idx + 1]):
                self.wall_proximity_near()
                self.update_Brownian_counter()
            #Current time is multiple of save time iteration
            elif self.adpt_dt_use and (np.round(self.curr_time,int(-np.log10(self.adpt_dt))) == self.all_time_vals[self.ar_idx + 1]):
                self.wall_proximity_far()
                self.reset_Brownian_counter()
                logging.info("Simulation is now exiting adaptive time stepping at t = {:.3e}".format(self.curr_time))
            #When filament is away from wall
            else:
                self.wall_proximity_far()
                self.reset_Brownian_counter()
             
        
    def wall_proximity_near(self):
        """
        This method changes the time-related parameters to be consistent with normal time stepping.
        """
        self.adpt_dt_use = True
        self.r1_use = self.adpt_r1
        self.r2_use = self.adpt_r2
        self.dt_use = self.adpt_dt
        self.zeta_use = self.adpt_zeta
    def wall_proximity_far(self):
        """
        This method changes the time-related parameters to be consistent with adaptive time stepping.
        """
        self.adpt_dt_use = False
        self.r1_use = self.true_r1
        self.r2_use = self.true_r2
        self.dt_use = self.true_dt
        self.zeta_use = self.true_zeta
    
    def update_Brownian_counter(self):
        """
        This method adds 1 to the Brownwian time counter during adaptive time stepping.
        """
        self.adpt_Brownian_counter += 1
        
        
    def reset_Brownian_counter(self):
        """
        This method resets the Brownian counter whenever the adaptive time steps are consistent with the normal time steps.
        """
        self.adpt_Brownian_counter = 0            

    def append_data(self,new_data):
        """
        This method appends the new or previous time steps to the list of arrays.
        """
        for i in range(0,18): #2D arrays
            self.all_data[i][:,:,self.ar_idx] = new_data[i]
        for i in range(18,20): #1D arrays
            self.all_data[i][:,self.ar_idx] = new_data[i]
        for i in range(20,24): #Singular values
            self.all_data[i][self.ar_idx] = new_data[i]
            
    
    def update_time(self):
        """
        This method either adds the adaptive or normal time steps to the current time.
        """
        if self.adpt_dt_use:
            self.add_adpt_time()
        else:
            self.add_true_time()
                    
    def add_iteration(self):
        """
        This method keeps track of the number of iterations needed to run the simulations.
        """
        self.it_count += 1
        self.record_it_count += 1
        
    def reset_record_iteration(self):
        """
        This method resets the iteration count.
        """
        self.record_it_count = 0

        
class run_parameters():
    """
    This class will write the important parameters that will be exported to a .CSV file
    in order to keep track of how to re-generate the data if needed.
    """
    def __init__(self,sim_params,end_time,start_time):
        """
        This initializes the class and writes all of the relevant parameters to a Pandas dataframe.
        """
        self.parameter_df = pd.DataFrame(index = ['Random Seed','Filament Length','Filament Persistence Length','Filament s start','Filament s end',
                                              'Poiseuille U Centerline','Kolmogorov Phase','Kolmogorov Frequency','Channel Upper Height','Channel Lower Height','Vertical Displacement',
                                              'Steric Velocity Exponential Coefficient','Steric Velocity Gap Criteria',
                                              'c','Mu_bar','True zeta','Adaptive zeta','N','theta_num','theta_den','True dt','Adaptive dt',
                                              'Array Time Step Size','Number of Iterations needed for Calculation','Simulation End Time',                                              
                                              'Calculation Time (sec)','Rigidity Function Type','Flow Type','Sterics Use','Brownian Use'],
                                    columns = ['Value'])

        self.parameter_df.loc['Random Seed','Value'] =sim_params.rand_seed
        self.parameter_df.loc['Filament Persistence Length','Value'] = sim_params.lp
        self.parameter_df.loc['Filament Length','Value'] = sim_params.L
        self.parameter_df.loc['Filament s start','Value'] = -sim_params.L/2
        self.parameter_df.loc['Filament s end','Value'] = sim_params.L/2
        self.parameter_df.loc['Poiseuille U Centerline'] = sim_params.U_centerline
        self.parameter_df.loc['Kolmogorov Phase'] = sim_params.kflow_phase
        self.parameter_df.loc['Kolmogorov Frequency'] = sim_params.kflow_freq
        self.parameter_df.loc['Channel Upper Height','Value'] = sim_params.channel_upper_height
        self.parameter_df.loc['Channel Lower Height','Value'] = sim_params.channel_lower_height
        self.parameter_df.loc['Vertical Displacement','Value'] = sim_params.vertical_displacement
        self.parameter_df.loc['Steric Velocity Exponential Coefficient','Value'] = sim_params.steric_velo_exp
        self.parameter_df.loc['Steric Velocity Gap Criteria','Value'] = sim_params.gap_criteria
        self.parameter_df.loc['c','Value'] = sim_params.c
        self.parameter_df.loc['Mu_bar','Value'] = sim_params.mu_bar
        self.parameter_df.loc['True zeta','Value'] = sim_params.true_zeta
        self.parameter_df.loc['Adaptive zeta','Value'] = sim_params.adpt_zeta
        self.parameter_df.loc['N','Value'] = sim_params.N
        self.parameter_df.loc['theta_num','Value'] = sim_params.numerator
        self.parameter_df.loc['theta_den','Value'] = sim_params.denominator
        self.parameter_df.loc['True dt','Value'] = sim_params.true_dt
        self.parameter_df.loc['Adaptive dt','Value'] = sim_params.adpt_dt
        self.parameter_df.loc['Array Time Step Size','Value'] = sim_params.save_time
        self.parameter_df.loc['Number of Iterations needed for Calculation','Value'] = sim_params.it_count
        self.parameter_df.loc['Simulation End Time','Value'] = sim_params.end_time
        self.parameter_df.loc['Flow Type','Value'] = 'Shear'
        self.parameter_df.loc['Sterics Use','Value'] = 'Yes'
        self.parameter_df.loc['Brownian Use','Value'] = 'Yes'
        if end_time != "FAIL" and start_time != 'FAIL':
            self.parameter_df.loc['Calculation Time (sec)','Value']  = end_time - start_time
        else:
            self.parameter_df.loc['Calculation Time (sec)','Value']  = "FAIL"
        self.parameter_df.loc['Rigidity Function Type','Value']  = sim_params.rigidity_suffix
        

        
        
def eval_time_semi(sim_c,prev_params,curr_params):
    """
    This function solves for position of the filament at the future time step using
    the semi-implicit method.
    
    Inputs:
    sim_c:          Class that contains all parameters needed for the simulation 
                    and arrays to store data.    
    prev_params:    Parameters of the previous time step needed to solve for 
                    the future filament position.
    curr_params:    Parameters of the current time step needed to solve for
                    the future filament position.
    t:              Current iteration of the simulation.
    """ 
    
    # Unpack relevant parameters to calculate explicit terms #
    x_loc_prev,prev_N,prev_xs = prev_params
    x_loc_curr,curr_N,curr_xs = curr_params
    
    ### Calculate component needed for dyadic ###
    adj_xs = 2*curr_xs - prev_xs
    
    ### Matrix construction ###
    lhs_matrix = construct_lhs_matrix(sim_c,adj_xs)
    rhs_matrix = ((2*x_loc_curr) + (-0.5 * x_loc_prev) + 
                  ((2*curr_N - prev_N)*sim_c.r2_use)).flatten()
    
    ### Force and torque-free boundary conditions ###
    rhs_matrix[0:6] = np.zeros(6,dtype = float)
    rhs_matrix[-6:] = np.zeros(6,dtype = float)
    
    ### Solve for future iteration ###
    try:
        future_xbar = np.linalg.solve(lhs_matrix,rhs_matrix).reshape(sim_c.N,3)
    except np.linalg.LinAlgError:
        logging.warning('\n')
        logging.warning(
            "Error: Instability & Singular Matrix detected.")
        logging.debug("Code will now stop at time = {:.8e} | iteration = {}".format(
                sim_c.curr_time,sim_c.it_count))
        logging.debug("Random seed that generated the instability: {}".format(sim_c.rand_seed))
                
        ### Save data to debugging CSV file ###
        failed_run_log_dir = os.path.join(sim_c.output_dir,'failed_logs/')
        create_dir(failed_run_log_dir)
        save_fail_params = run_parameters(sim_c,"FAIL", "FAIL")
        failed_run_logs_all = glob.glob(os.path.join(failed_run_log_dir,'failed_run_log*.csv'))
        if failed_run_logs_all:
            all_nums = []
            for l_file in failed_run_logs_all:
                match = re.search(r"failed_run_log_(\d{2}).csv",l_file)
                if match:
                    file_num = int(match.group(1))
                    all_nums.append(file_num)
            new_file_num = np.array(all_nums).max() + 1
        else:
            new_file_num = 0
        save_fail_params.parameter_df.to_csv(os.path.join(failed_run_log_dir,'failed_run_log_{:02d}.csv'.format(new_file_num)))
        
        ### Check if script needs to restart ###
        if sim_c.retry_type:
            logging.critical("Code will now restart from the beginning.")
            os.execl(sys.executable,os.path.abspath(__file__),*sys.argv)
        else:
            logging.critical("Code has not halted.")
            sys.exit(1)
        
    fil_length = rod_length(sim_c.N,future_xbar)
    ##### Calculate parameters at next iteration #####

    ### Determine which time step value to use to calculate Brownian motion ###
    sim_c.update_time()
    sim_c.det_proximity_wall(future_xbar)
    sim_c.add_iteration()
    curr_time = sim_c.curr_time  
    ### Determine fluid velocity and derivative
    u0,du0ds = fluid_velo(sim_c,future_xbar)
    
    ### Calculate spatial derivative ###
    dxds, d2xds2, d3xds3, d4xds4 = spatialderiv(sim_c,future_xbar)
    
    ### Calculate Brownian forces & check when to use for adaptive time step ###
    if sim_c.adpt_dt_use and sim_c.adpt_Brownian_counter == sim_c.adpt_dt_Brownian_ratio:
        # f_brownian = np.zeros((sim_c.N,3),dtype = float)
        f_brownian = calc_Brownian(sim_c,dxds)
        sim_c.reset_Brownian_counter()
    elif sim_c.adpt_dt_use and not sim_c.adpt_Brownian_counter == sim_c.adpt_dt_Brownian_ratio:
        f_brownian = np.zeros((sim_c.N,3),dtype = float)
    elif not sim_c.adpt_dt_use:
        f_brownian = calc_Brownian(sim_c,dxds)
    # f_brownian = calc_Brownian(sim_c,dxds)
    
    ### Caculate Tension and its derivative ###
    tension = solve_tension(sim_c,du0ds, dxds, d2xds2, d3xds3, d4xds4,f_brownian)
    dTds = first_deriv(tension,sim_c.N,sim_c.ds,1)
    
    #Obtain spatial-derivative coupled terms
    fut_params = calc_vals(sim_c,tension, dTds, dxds, d2xds2, d3xds3, d4xds4)
    Txss,Tsxs,Kxs,Kxssss,Ksxs,Ksxsss,Kssxss = fut_params  
    force = calc_force(sim_c,Txss,Tsxs,Kssxss,Ksxsss,Kxssss,f_brownian,'all')
    stress = calc_stress(sim_c,force, future_xbar)
    elastic_energy = calc_E_elastic(sim_c,d2xds2)
    sim_c,fil_angle,fil_deflect = det_rotation(sim_c,future_xbar)
    
    ### Calculate non-time and elastic velocities for steric velocity ###
    steric_velocity_params = [u0, du0ds, dxds, d2xds2, d3xds3,
    Txss,Tsxs,Kxs,Ksxs,Ksxsss,Kssxss,f_brownian]
    expl_U_N_ex = calculate_N_ex(sim_c, steric_velocity_params)
    
    #Calculate corrected non-elastic velocity due to confinement
    corrected_U_N_ex = steric_velocity(sim_c,future_xbar,expl_U_N_ex)
    # corrected_U_N_ex = expl_U_N_ex.copy()
    
    var_to_pack = []
    var_to_pack = [future_xbar,corrected_U_N_ex,u0,du0ds,dxds, d2xds2, #0-5
                   d3xds3, d4xds4,Txss,Tsxs,Kxs, #6-10
                   Kxssss,Ksxs,Ksxsss,Kssxss,f_brownian, #11-15
                   force,stress, #16-17
                   tension,dTds, #18-19
                   elastic_energy,fil_length,fil_angle,fil_deflect] #20-23
    
    #Record data every save time iteration
    if np.round(curr_time,int(-np.log10(sim_c.adpt_dt))) == sim_c.all_time_vals[sim_c.ar_idx + 1]:
        #Record variables into arrays
        sim_c.add_ar_idx()
        sim_c.reset_record_iteration()
        sim_c.append_data(var_to_pack)
    prev_params = curr_params
    curr_params = [v for i,v in enumerate(var_to_pack[:5]) if i != 2 and i != 3]
        
    return sim_c,prev_params,curr_params,var_to_pack


def eval_time_euler(sim_c):
    """
    This function solves for position of the filament at the future time step using a
    general Euler method.
    
    Inputs:
    sim_c:      Class that contains all parameters needed for the simulation 
                and arrays to store data.
    t:          Current iteration of the simulation.
    """  
    
    ### Check if filament is within proximity to wall to use adaptive time step ###
    sim_c.det_proximity_wall(sim_c.initial_loc)
    
    ### Get filament length ###
    fil_length = rod_length(sim_c.N,sim_c.initial_loc)
    ### Determine fluid velocity and derivative ###
    u0,du0ds = fluid_velo(sim_c,sim_c.initial_loc)
    
    ### Calculate spatial derivative ###
    dxds, d2xds2, d3xds3, d4xds4 = spatialderiv(sim_c,sim_c.initial_loc)
    
    ### Calculate Brownian forces ###
    # f_brownian = np.zeros((sim_c.N,3),dtype = float)
    f_brownian = calc_Brownian(sim_c,dxds)

    ### Caculate Tension and its derivative ###
    tension = solve_tension(sim_c,du0ds, dxds, d2xds2, d3xds3, d4xds4,f_brownian)
    dTds = first_deriv(tension,sim_c.N,sim_c.ds,1)
    
    #Obtain spatial-derivative coupled terms
    curr_params = calc_vals(sim_c,tension, dTds, dxds, d2xds2, d3xds3, d4xds4)
    Txss,Tsxs,Kxs,Kxssss,Ksxs,Ksxsss,Kssxss = curr_params    
    force = calc_force(sim_c,Txss,Tsxs,Kssxss,Ksxsss,Kxssss,f_brownian,'all')
    stress = calc_stress(sim_c,force, sim_c.initial_loc)
    elastic_energy = calc_E_elastic(sim_c,d2xds2)
    sim_c,fil_angle,fil_deflect = det_rotation(sim_c,sim_c.initial_loc)
    
    
    ### Calculate non-time and elastic velocities for steric velocity ###
    steric_velocity_params = [u0, du0ds, dxds, d2xds2, d3xds3,
    Txss,Tsxs,Kxs,Ksxs,Ksxsss,Kssxss,f_brownian]
    expl_U_N_ex = calculate_N_ex(sim_c, steric_velocity_params)
    
    #Calculate corrected non-elastic velocity due to confinement
    corrected_U_N_ex = steric_velocity(sim_c,sim_c.initial_loc,expl_U_N_ex)
    # corrected_U_N_ex = expl_U_N_ex.copy()
    
    #Record variables into arrays
    var_to_pack = []
    var_to_pack = [sim_c.initial_loc,corrected_U_N_ex,u0,du0ds,dxds, d2xds2, #0-5
                   d3xds3, d4xds4,Txss,Tsxs,Kxs, #6-10
                   Kxssss,Ksxs,Ksxsss,Kssxss,f_brownian, #11-15
                   force, stress, #16-17
                   tension,dTds, #18-19
                   elastic_energy,fil_length,fil_angle,fil_deflect] #20-23
    
    #Save initial data
    sim_c.append_data(var_to_pack)     
    mu_bar,c,N = sim_c.mu_bar,sim_c.c,sim_c.N
    
    ##### Solve for next iteration #####
    
    xt = (mu_bar*u0 -(c+1)*force-(c-3)*(Kxs*np.repeat(rdot(
        dxds,d4xds4),3).reshape(N,3) + \
            2*Ksxs*np.repeat(rdot(dxds,d3xds3),3).reshape(N,3) - Tsxs))
                
    xt[0,:] = (1/11)*(48*xt[2,:] - 52*xt[3,:] + 15*xt[4,:])
    xt[1,:] = (1/11)*(28*xt[2,:] - 23*xt[3,:] + 6*xt[4,:])
    xt[N-1,:] = (1/11)*(48*xt[N-3,:] - 52*xt[N-4,:] + 15*xt[N-5,:])
    xt[N-2,:] = (1/11)*(28*xt[N-3,:] - 23*xt[N-4,:] + 6*xt[N-5,:])

    future_xbar = sim_c.initial_loc + 1e-5*sim_c.dt_use*xt #filament position at incremental small time step
    fil_length_1 = rod_length(sim_c.N,future_xbar)
    
    ### Check if filament is within proximity to wall to use adaptive time step ###
    sim_c.det_proximity_wall(future_xbar)
        
    ### Determine fluid velocity and derivative
    u0_1,du0ds_1 = fluid_velo(sim_c,future_xbar)
    
    ### Calculate spatial derivative
    dxds_1, d2xds2_1, d3xds3_1, d4xds4_1 = spatialderiv(sim_c,future_xbar)

    ### Calculate Brownian forces ###
    # f_brownian_1 = np.zeros((sim_c.N,3),dtype = float)
    f_brownian_1 = calc_Brownian(sim_c,dxds_1)
    
    ### Caculate Tension and its derivative ###
    tension_1 = solve_tension(sim_c,du0ds_1, dxds_1, d2xds2_1, d3xds3_1, d4xds4_1,f_brownian_1)
    dTds_1 = first_deriv(tension_1,sim_c.N,sim_c.ds,1)
    
    #Obtain spatial-derivative coupled terms
    fut_params = calc_vals(sim_c,tension_1, dTds_1, dxds_1, d2xds2_1, d3xds3_1, d4xds4_1)
    Txss_1,Tsxs_1,Kxs_1,Kxssss_1,Ksxs_1,Ksxsss_1,Kssxss_1 = fut_params    
    force_1 = calc_force(sim_c,Txss_1,Tsxs_1,Kssxss_1,Ksxsss_1,Kxssss_1,f_brownian_1,'all')
    stress_1 = calc_stress(sim_c,force_1, future_xbar)
    elastic_energy_1 = calc_E_elastic(sim_c,d2xds2_1)
    sim_c,fil_angle_1,fil_deflect_1 = det_rotation(sim_c,future_xbar)
    
    ### Calculate non-time and elastic velocities for steric velocity ###
    steric_velocity_params_1 = [u0_1, du0ds_1,  dxds_1, d2xds2_1, d3xds3_1,
    Txss_1,Tsxs_1,Kxs_1,Ksxs_1,Ksxsss_1,Kssxss_1,f_brownian_1]
    expl_U_N_ex_1 = calculate_N_ex(sim_c, steric_velocity_params_1)
    
    #Calculate corrected non-elastic velocity due to confinement
    corrected_U_N_ex_1 = steric_velocity(sim_c,future_xbar,expl_U_N_ex_1)
    # corrected_U_N_ex_1 = expl_U_N_ex_1.copy()
    
    #Save data for next iteration
    fut_var_to_pack = []
    fut_var_to_pack = [future_xbar,corrected_U_N_ex_1,u0_1,du0ds_1,dxds_1, d2xds2_1, #0-5
                   d3xds3_1, d4xds4_1,Txss_1,Tsxs_1,Kxs_1, #6-10
                   Kxssss_1,Ksxs_1,Ksxsss_1,Kssxss_1,f_brownian_1, #11-15
                   force_1,stress_1, #16-17
                   tension_1,dTds_1, #18-19
                   elastic_energy_1,fil_length_1,fil_angle_1,fil_deflect_1] #20-23
    
    return sim_c,fut_var_to_pack
            

#%% if__name__ == '__main__' Method
def run_simulation(target_dir,rigid_type,mu_bar,rep):
    """
    This function runs the simulation for the filament at a specified mu_bar value.
    
    Inputs:
    target_dir:     Main output directory where the simulations files & folders
                    will reside in.
    rigid_type:     Type of filament rigidity that the simulations will run on.
    mu_bar:         Mu_bar value to run the simulation at.
    rep:            Replicate number. 
    """
    
    ####################### Initialize all parameters an`d output directory #######################
    sim_c = Constants(end_dir = target_dir,rigid_func = rigid_type,
                                slender_f = 0.01,mubar = mu_bar,true_zeta_f =3e6,adpt_zeta_f = 6e7,
                                top = 0, bottom = 1,
                                U_centerline = 0,k_flow_phase = 0,k_flow_factor = 0,
                                gap_criteria = 0.02,force_power = 12,channel_height = 0.50,
                                vert_displace = 0.15,N = 101,lp = 1000,reg_dt = 1e-7,adpt_dt = 1e-8,
                                end_time = 5e-2,save_time = 1e-5)
    vert_displ_str = '{:.2f}'.format(sim_c.vertical_displacement).replace('.','p')
    H_val_str = '{:.2f}'.format(sim_c.channel_upper_height).replace('.','p')
    dir_name = 'VD_{}/{}_H_{}/MB{}/R_{}'.format(vert_displ_str,sim_c.rigidity_suffix,H_val_str,
                                                                int(sim_c.mu_bar),rep)
    output_dir = os.path.join(sim_c.output_dir,dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.info("Resulting data will be saved to {}.".format(output_dir))
    sim_c.output_dir = output_dir
    ####################### Time iterations-Initial Euler Step #######################
        
    start_time = time.perf_counter()
    sim_c,small_step_vars = eval_time_euler(sim_c)
    prev_info = [sim_c.all_data[i][:,:,0] for i in range(0,5) if i != 2 and i != 3]#  unpack parameters at t = 0 #
    curr_info = [small_step_vars[i] for i in range(0,5) if i != 2 and i != 3] # unpack parameters at next iteration

    ####################### Time iterations-Subsequent Steps #######################
    while sim_c.curr_time < sim_c.end_time:
        sim_c,prev_info,curr_info,all_vars = eval_time_semi(sim_c,prev_info,curr_info)
        if sim_c.record_it_count == 0:
            if np.isnan(sim_c.all_data[0][:,:,sim_c.ar_idx]).any():
                logging.warning("\n")
                
                logging.warning(
                    "Error: Filament positional data is invalid.")
                logging.debug("Code will now stop at time = {:.8e} | iteration = {}".format(
                        sim_c.curr_time,sim_c.it_count))
                logging.debug("Random seed that generated the instability: {}".format(sim_c.rand_seed))
                
                ### Save data to debugging CSV file ###
                failed_run_log_dir = os.path.join(target_dir,'failed_logs/')
                create_dir(failed_run_log_dir)
                save_fail_params = run_parameters(sim_c,"FAIL", "FAIL")
                failed_run_logs_all = glob.glob(os.path.join(failed_run_log_dir,'failed_run_log*.csv'))
                if failed_run_logs_all:
                    all_nums = []
                    for l_file in failed_run_logs_all:
                        match = re.search(r"failed_run_log_(\d{2}).csv",l_file)
                        if match:
                            file_num = int(match.group(1))
                            all_nums.append(file_num)
                    new_file_num = np.array(all_nums).max() + 1
                else:
                    new_file_num = 0
                save_fail_params.parameter_df.to_csv(os.path.join(failed_run_log_dir,'failed_run_log_{:02d}.csv'.format(new_file_num)))
                
                ### Check if script needs to restart ###
                if sim_c.retry_type:
                    logging.critical("Code will now restart from the beginning.")
                    os.execl(sys.executable,os.path.abspath(__file__),*sys.argv)
                else:
                    logging.critical("Code has not halted.")
                    sys.exit(1)
                    
        if not sim_c.it_count % 100000:
                logging.info("Simulation is now at t = {:.5e}".format(sim_c.curr_time))

    end_time = time.perf_counter()
    logging.info("Finished with the numerical simulations calculations portion for Mu_bar = {}, Replicate # {}. Data will be written to {}."\
                  .format(mu_bar,rep,output_dir))

    ### Save Run Parameters ###
    save_sim_params = run_parameters(sim_c, end_time, start_time)

    ### Save Data ###
    np.save(os.path.join(output_dir,'filament_allstate.npy'),sim_c.all_data[0])
    np.save(os.path.join(output_dir,'filament_stress_all.npy'),sim_c.all_data[17])
    np.save(os.path.join(output_dir,'filament_tension.npy'),sim_c.all_data[18])
    np.save(os.path.join(output_dir,'filament_elastic_energy.npy'),sim_c.all_data[20])
    np.save(os.path.join(output_dir,'filament_length.npy'),sim_c.all_data[21])
    np.save(os.path.join(output_dir,'filament_angles.npy'),sim_c.all_data[22])
    np.save(os.path.join(output_dir,'filament_deflection.npy'),sim_c.all_data[23])
    np.save(os.path.join(output_dir,'filament_time_vals_all.npy'),sim_c.all_time_vals)
    save_sim_params.parameter_df.to_csv(os.path.join(output_dir,'parameter_values.csv'))
    
    return sim_c,save_sim_params
    
def Main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_directory", 
                        help="The main directory where the simulation files will reside in",
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
    parser.add_argument("replicate_value",
                        help = "Specify what the replicate number will be in the numbering scheme",
                        type = int)
    # args = parser.parse_args(['./01_Test_Results/U_225_Param_Test/' ,'K_constant', '20000','1']) #Uncommment this line when debugging
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s|%(filename)s|%(levelname)s|%(message)s',
            datefmt="%A, %B %d at %I:%M:%S %p")
    logging.info(
        "Started Simulation script for a rigidity profile of {},Mu_bar = {}, Replicate # {}".format(
            args.rigidity_type,args.mu_bar,args.replicate_value))
                 
    start_time = time.perf_counter()
    sim_c,save_sim_params = run_simulation(args.output_directory,args.rigidity_type,args.mu_bar,args.replicate_value)
    end_time = time.perf_counter()
    logging.info("Finished all computations. Time to complete all tasks is {} seconds".format(
            end_time - start_time))
    return sim_c,save_sim_params

# ####################### Initialization of Main Script #######################
if __name__ == '__main__':
    __spec__ = None
    sim_data,sim_params_all = Main()
    

#%% ####################### Post-Visualization: Filament Length #######################

# plt.figure(figsize = (8,8))
# plt.plot(np.linspace(0,sim_data.end_time,sim_data.adj_iterations),sim_data.all_data[21])

# plt.axis()
# ax = plt.gca()
# plt.xlabel('Time',fontsize=16)
# plt.ylabel('Filament length',fontsize=16)
# ax.set_title('Filament Length over Simulation')
# # plt.ylim(0.999,1.001)
# plt.show()



#%% ####################### Post-Visualization: Filament Tension #######################


# plt.figure(figsize = (8,8))
# plt.plot(sim_data.all_data[18][:,0],'b',label = 'Initial Tension')
# plt.plot(sim_data.all_data[18][:,-1],'r',label = 'Final Tension')

# plt.axis()
# ax = plt.gca()
# plt.ylabel('Tension',fontsize=16)
# # plt.ylim(0.999,1.001)

# plt.show()
#%% ####################### Post-Visualization: Filament Positions #######################

### Initial Filament Position
# plt.figure(figsize = (8,8))
# plt.plot(sim_data.all_data[0][:,0,0],sim_data.all_data[0][:,1,0],'b',label = 'Initial Position')
# plt.plot(sim_data.all_data[0][:,0,11] - sim_data.all_data[0][:,0,11].mean(),sim_data.all_data[0][:,1,11],'magenta',label = 'Desired Position')
# plt.plot(sim_data.all_data[0][:,0,sim_data.ar_idx] - sim_data.all_data[0][:,0,sim_data.ar_idx].mean(),sim_data.all_data[0][:,1,sim_data.ar_idx],'r',label = 'Final Position')
# plt.axis('square')
# plt.xlabel('x',fontsize=20)
# plt.ylabel('y',fontsize=20)
# plt.legend(['Initial Pos','Buckling','Final Pos'],fontsize=20,loc='upper right')
# plt.legend(fontsize=20,loc='upper right')
# plt.xlim(-0.6,0.6)
# plt.ylim(-0.6,0.6)

# plt.show()

