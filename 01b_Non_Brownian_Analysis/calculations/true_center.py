# -*- coding: utf-8 -*-
"""
FILE NAME:      true_center.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    This function calculates the filament true center provided
                a Nx3xT position array.
INPUT
FILES(S):       N/A

OUTPUT
FILES(S):       N/A


INPUT
ARGUMENT(S):                

position_data:              Nx3xT filament position array.
center_index:               Array index that indicates the true center of the filament.
y_translation:              Boolean varaible to specify whether  or not to adjust
                            the filament y-values based on a specified value.
transl_val:                 Value to adjust the filament y-values by; will be subtracted.
                                             


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
import os,argparse
import numpy as np

def true_center(position_data,center_idx,adj_translation,transl_val):
    """
    This function calculates the true center of the filament along a specific
    position (x,y,z) and axis. As this function currently stands, the position data
    is a Nx3xT array where N is the total number of points along the filament, and
    T is the number of time points calculated for a particular simulation. The second 
    axis are the position axes. This function can also adjust the x-coordinates
    so that the filament true center-x is centered at x=0. Additionally, the
    y-coordinates can be adjusted by a value as well. The center of 
    mass will be calculated along the 0th array (N). The resulting array, will 
    have rows that represent the coordinates at each timepoint and the columns 
    represent the x,y,z positions. 
    
    Inputs:
        
    position_data:              Nx3xT filament position array.
    center_idx:                 The array index that indicates the true center
                                of the filament.
    adj_translation:            Boolean varaible to specify whether or not to
                                adjust the filament y-values based on a specified
                                value.
    transl_val:                 Value to adjust the filament y-values by; will
                                be subtracted.
    """
    true_center_vals = position_data[center_idx,:,:].copy().T #Rows are time points, columns are x,y,z components
    if adj_translation:
        true_center_vals[:,1] -= transl_val
    return true_center_vals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("array_path", 
                        help="The absolute path to the .NPY position file to calculate the true center of",
                    type = str)
    parser.add_argument("center_index",
                        help = "Specify the array index that indiates the true center of the filament",
                        type = int)
    parser.add_argument("--y_translation",'-yt',
                        help = 'Specify whether or not you want the filament y-coordinates to be adjusted by a value',
                        action = 'store_true',default = False)
    parser.add_argument("--y_translation_val",'-ytv',
                        help = 'Value to adjust the y-coordinates of the filament by',
                        type = float,default = 0)
    args = parser.parse_args()
    
    true_center(position_data = np.load(args.array_path),
                center_idx = args.center_idx, adj_translation = args.y_translation,
                   transl_val = args.y_translation_val)