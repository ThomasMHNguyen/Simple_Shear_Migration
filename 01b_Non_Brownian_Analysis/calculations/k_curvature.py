# -*- coding: utf-8 -*-
"""
FILE NAME:      k_curvature.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    This function calculates the curvature of the filament based on the first
                and second spatial derivatives of filament position. Refer to 
                https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization
                for the exact formula.

INPUT
FILES(S):       1) .NPY file that contains information about the filament's first
                spatial derivative at every point along the filament and every
                timepoint.
                2) .NPY file that contains information about the filament's first
                spatial derivative at every point along the filament and every
                timepoint.

OUTPUT
FILES(S):       N/A

INPUT
ARGUMENT(S):    1) Project Directory: The directory that contains all of the 
                complementary scripts needed to run the analysis.
                2) First Derivative Array Path: The absolute path to the .NPY
                file that contains information about the first spatial derivative
                of the filament at every point along it and every timepoint.
                3) Main Input shear directory: The absolute path to the .NPY
                file that contains information about the second spatial derivative
                of the filament at every point along it and every timepoint.


CREATED:        14Jun23

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

import argparse, os
import numpy as np


def k_curvature(xs,xss):
    """
    This function calculates the curvature of the filament based on the first
    and second spatial derivatives of filament position.
    
    Inputs:
        
    xs:             Nx3xT array that contains the first derivative information
                    at every point on the filament (N) at every timepoint (T).
    xss:            Nx3xT array that contains the second derivative information
                    at every point on the filament (N) at every timepoint (T).
    """
    
    k_curv = (np.abs((xs[:,0,:]*xss[:,1,:]) - (xs[:,1,:]*xss[:,0,:])))/\
    (((xs[:,0,:]**2)+(xs[:,1,:]**2))**(1.5))
    return k_curv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_directory",
                        help="Specify the directory where the main script and accompany scripts are located in",
                    type = str)
    parser.add_argument("f_derivative_path",
                        help="Specify the absolute path to the numpy array file (.NPY) that contains the first derivative (x_s) information",
                    type = str)
    parser.add_argument("s_derivative_path",
                        help="Specify the absolute path to the numpy array file (.NPY) that contains the second derivative (x_ss) information",
                    type = str)
    args = parser.parse_args()
    
    os.chdir(args.project_directory)
    
    ### Calculate filament curvature data ###
    xs = np.load(args.f_derivative_path)
    xss = np.load(args.s_derivative_path)
    k_curv = k_curvature(xs,xss)