# -*- coding: utf-8 -*-
"""
FILE NAME:      create_dir.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        N/A

DESCRIPTION:    This script will create a directory if it doesn't already exist.

INPUT
FILES(S):       N/

OUTPUT
FILES(S):       1) N/A


INPUT
ARGUMENT(S):    1) Directory: The directory will be created if it doesn't 'exist.


CREATED:        06Feb23

MODIFICATIONS
LOG:            N/A

LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.8.8

VERSION:        1.0

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:    N/A

NOTE(S):       N/A

"""
import os, time, argparse


def create_dir(dir_):
    """
    This function creates the directory if it doesn't exist.

    Inputs:
        
    dir_:           Absolute path to the directory that will be created.

    """
    if not os.path.exists(dir_):
        while True:
            try:
                os.makedirs(dir_)
                break
            except OSError:
                continue
                time.sleep(1)
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("Directory", 
                        help="The directory that will be created if it doesn't exist",
                    type = str)
    args = parser.parse_args()
    create_dir(args.directory)