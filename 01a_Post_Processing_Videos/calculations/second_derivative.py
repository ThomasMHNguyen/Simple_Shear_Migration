# -*- coding: utf-8 -*-

import re, sys, os, argparse
import numpy as np


def second_derivative(base_array,deriv_size,axis,ar_size,dim):
    """
    This function calculates the second derivative of an array.
    
    Inputs:
    base_array:             The array to calculate the second derivative of.
    deriv_size:             The first derivative of an array will be calculated
                            with respect to a variable; this is the size of that
                            particular variable.
    axis:                   The array axis to take the derivative along.
    ar_size:                The size of the array to calculate the first 
                            derivative along.
    dim:                    Dimension of the array
    """
    
    if dim == 2:
        d2ds2_base = np.zeros((base_array.shape),dtype = float)
        if axis == 0:
            d2ds2_base[0,:,:] = (1/deriv_size**2)*(2*base_array[0,:,:]-\
                                                   5*base_array[1,:,:]+\
                                                       4*base_array[2,:,:]-\
                                                           base_array[3,:,:])
            d2ds2_base[ar_size-1,:,:] = (1/deriv_size**2)*(2*base_array[ar_size-1,:,:]-\
                                                           5*base_array[ar_size-2,:,:]+\
                                                               4*base_array[ar_size-3,:,:]-\
                                                                   base_array[ar_size-4,:,:])
            d2ds2_base[1:ar_size-1,:,:] = (1/deriv_size**2)*(base_array[2:ar_size,:,:]-\
                                                             2*base_array[1:ar_size-1,:,:]+\
                                                                 base_array[0:ar_size-2,:,:])
        elif axis == 1:
            d2ds2_base[:,0,:] = (1/deriv_size**2)*(2*base_array[:,0,:]-\
                                                   5*base_array[:,1,:]+\
                                                       4*base_array[:,2,:]-\
                                                           base_array[:,3,:])
            d2ds2_base[:,ar_size-1,:] = (1/deriv_size**2)*(2*base_array[:,ar_size-1,:]-\
                                                           5*base_array[:,ar_size-2,:]+\
                                                               4*base_array[:,ar_size-3,:]-\
                                                                   base_array[:,ar_size-4,:])
            d2ds2_base[:,1:ar_size-1,:] = (1/deriv_size**2)*(base_array[:,2:ar_size,:]-\
                                                             2*base_array[:,1:ar_size-1,:]+\
                                                                 base_array[:,0:ar_size-2,:])
        elif axis == 2:
            d2ds2_base[:,:,0] = (1/deriv_size**2)*(2*base_array[:,:,0]-\
                                                   5*base_array[:,:,1]+\
                                                       4*base_array[:,:,2]-\
                                                           base_array[:,:,3])
            d2ds2_base[:,:,ar_size-1] = (1/deriv_size**2)*(2*base_array[:,:,ar_size-1]-\
                                                           5*base_array[:,:,ar_size-2]+\
                                                               4*base_array[:,:,ar_size-3]-\
                                                                   base_array[:,:,ar_size-4])
            d2ds2_base[:,:,1:ar_size-1] = (1/deriv_size**2)*(base_array[:,:,2:ar_size]-\
                                                             2*base_array[:,:,1:ar_size-1]+\
                                                                 base_array[:,:,0:ar_size-2])
    return d2ds2_base

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("array_path", 
                        help="The absolute path to the .NPY file to calculate the first derivative of",
                    type = str)
    parser.add_argument("derivative_size",
                        help = 'The first derivative of the specified array will be calculated with respect to a variable; this is the size of that variable',
                        type = float)
    parser.add_argument("-array_size","--ars",
                        help = "Specify the size of the array to calculate the first derivative along",
                        type = float,required = False)
    args = parser.parse_args()
    
    array = np.load(args.array_path)
    if not args.array_size:
        array_size = array.shape[0]
    else:
        array_size = args.array_size
    
    array_dim = array.shape[0]
    
    second_derivative(array,args.derivative_size,array_size,array_dim)