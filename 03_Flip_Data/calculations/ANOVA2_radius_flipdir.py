"""
FILE NAME:              ANOVA2_radius_flipdir.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):                Process_Filament_Flip_Data.py

DESCRIPTION:            This script cuntions a function that performs a 2-way ANOVA to see 
                        if there's a significant relationship between radius of bending due to 
                        drift, mu_bar, and displacement type. After performing the ANOVA,
                        it will perform an ad-hoc Tukey test as an FDR to see which relationships
                        are significant.
                       

INPUT
FILES(S):               N/A

OUTPUT
FILES(S):               N?A


INPUT
ARGUMENT(S):            N/A

CREATED:                22Nov22

MODIFICATIONS
LOG:
22Nov22                 1) Migrated code to generate the plots from the original script
                        to its own instance here.
26Sep23                 2) Separated net drift and radius ANOVA functions into separate scripts. 

            
LAST MODIFIED
BY:                     Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:                 3.8.8

VERSION:                1.0

AUTHOR(S):              Thomas Nguyen

STATUS:                 Working

TO DO LIST:             N/A

NOTE(S):                N/A

"""

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def ANOVA2_radius_flipdir(input_file,output_directory,file_name):
    """
    This function performs a 2-way ANOVA to see if there's a significant relationship
    between radius of bending, mu_bar, and displacement type. After
    performing the ANOVA, it will perform an ad-hoc Tukey test as an FDR to see
    which relationships are significant.
    
    Inputs:
    
    input_file:                 Input Pandas dataframe that lists the measured
                                data values for each U-turn flipping event.
    output_directory:           Path to directory to be created.
    file_name:                  The file name to be used for the output graphical
                                files.
    """
    
    ### Perform 2-way ANOVA ###
    
    input_file['Mu_bar_Displacement_Type'] = input_file['Mu_bar String'].astype(str) + '|' + input_file['Displacement Type']
    model = smf.ols("Q('Radius of Bending-1') ~ Q('Displacement Type') + Q('Mu_bar String') + Q('Displacement Type') :Q('Mu_bar String')",data = input_file).fit()
    sm.stats.anova_lm(model, typ=2)
    
    ### Perform Ad hoc corrections for 2-way ANOVA ###
    m_comp = pairwise_tukeyhsd(endog=input_file['ABS Net COM-y'], groups=input_file['Mu_bar_Displacement_Type'], alpha=0.05)
    tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:], columns = m_comp._results_table.data[0])
    tukey_data['Mu_bar_1'] = tukey_data['group1'].apply(lambda x: x.split('|')[0])
    tukey_data['Displacement_Type_1'] = tukey_data['group1'].apply(lambda x: x.split('|')[1])
    tukey_data['Mu_bar_2'] = tukey_data['group2'].apply(lambda x: x.split('|')[0])
    tukey_data['Displacement_Type_2'] = tukey_data['group2'].apply(lambda x: x.split('|')[1])
    tukey_data = tukey_data[tukey_data['Mu_bar_1'] == tukey_data['Mu_bar_2']]
    
    new_file_name = '{}.csv'.format(file_name)
    tukey_data.to_csv(os.path.join(output_directory,new_file_name))
    return tukey_data