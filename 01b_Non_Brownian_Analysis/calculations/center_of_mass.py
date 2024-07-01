# -*- coding: utf-8 -*-
"""
FILE NAME:                  center_of_mass.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):                    N/A

DESCRIPTION:                This function calculates the filament center of mass provided
                            a Nx3xT position array.
INPUT
FILES(S):                   1) .NPY file that contains information about the 
                            filament position data along every coordinate axis
                            and every timepoint.

OUTPUT
FILES(S):                   N/A


INPUT
ARGUMENT(S):

array_path:                 Nx3xT filament position array.
axis_average:               Axis of the filament to calculate the center of mass along.
array_dimension:            Dimension of the array (needs to be 3).
x_centering:                Boolean variable to specify whether to center
                            the filament at x = 0.
y_translation:              Boolean varaible to specify whether  or not to adjust
                            the filament y-values based on a specified value.
transl_val:                 Value to adjust the filament y-values by; will be subtracted.
                                             


CREATED:                    05Apr23

MODIFICATIONS
LOG:                        N/A

            
LAST MODIFIED
BY:                         Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:                     3.9.13

VERSION:                    1.0

AUTHOR(S):                  Thomas Nguyen

STATUS:                     Working

TO DO LIST:                 N/A

NOTE(S):                    N/A

"""
import os,argparse
import numpy as np

def center_of_mass(position_data,position,dim,adj_centering,adj_translation,transl_val):
    """
    This function calculates the center of mass of the filament along a specific
    position (x,y,z) and axis. As this function currently stands, the position data
    is a Nx3xT array where N is the total number of points along the filament, and
    T is the number of time points calculated for a particular simulation. The second 
    axis are the position axes. This function can also adjust the x-coordinates
    so that the filament center of mass-x is centered at x=0. Additionally, the
    y-coordinates can be adjusted by a value as well. The center of 
    mass will be calculated along the 0th array (N). The resulting array, will 
    have rows that represent the coordinates at each timepoint and the columns 
    represent the x,y,z positions. 
    
    Inputs:
        
    position_data:              Nx3xT filament position array.
    position:                   Axis of the filament to calculate the center of
                                mass along.
    dim:                        Dimension of the array (needs to be 3).
    adj_centering:              Boolean variable to specify whether to center
                                the center the filament at x = 0.
    adj_translation:            Boolean varaible to specify whether or not to
                                adjust the filament y-values based on a specified
                                value.
    transl_val:                 Value to adjust the filament y-values by; will
                                be subtracted.
    """
    position_data_c = position_data.copy()
    if dim == 3:
        if position == 0:
            if adj_centering:
                position_data_c[:,0,:] = position_data_c[:,0,:] - \
                    position_data_c[:,0,:].mean(axis = 0)#Adjust for any translation in x-coordinates
            if adj_translation:
                position_data_c[:,1,:] = position_data_c[:,1,:] - transl_val
                
            center_of_mass = position_data_c.mean(axis = position).copy().T #Rows are each time point, columns are x,y,z, components
    return center_of_mass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("array_path", 
                        help="The absolute path to the .NPY position file to calculate the center of mass of",
                    type = str)
    parser.add_argument("axis_average",
                        help = 'The array axis to calculate the center of mass on',
                        type = int)
    parser.add_argument("array_dimension",
                        help = 'The dimensionality of the array',
                        type = int)
    parser.add_argument("--x_centering",'-xc',
                        help = 'Specify whether or not you want the filament to be centered at x= 0',
                        action = 'store_true',default = False)
    parser.add_argument("--y_translation",'-yt',
                        help = 'Specify whether or not you want the filament y-coordinates to be adjusted by a value',
                        action = 'store_true',default = False)
    parser.add_argument("--y_translation_val",'-ytv',
                        help = 'Value to adjust the y-coordinates of the filament by',
                        type = float,default = 0)
    args = parser.parse_args()
    
    center_of_mass(position_data = np.load(args.array_path),
                   position = args.axis_average, dim = args.array_dimension,
                   adj_centering = args.x_centering, adj_translation = args.y_translation,
                   transl_val = args.y_translation_val)