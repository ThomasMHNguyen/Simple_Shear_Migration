# -*- coding: utf-8 -*-
"""
FILE NAME:          ANOVA2_netycom_phase_flipdir.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):            Compare_Poiseuille_Shear_Data.py

DESCRIPTION:        This script performs a 2-way ANOVA to see if there's a 
                    significant relationship between net displacement due to 
                    drift, phase of the transition, and Displacement Type. 
                    After performing the ANOVA, it will perform an ad-hoc Tukey 
                    test as an FDR to see which relationships are significant.

INPUT
FILES(S):           N/A

OUTPUT
FILES(S):       
    
1)                  .PNG/.PDF/.EPS of a violin plot that shows the net center of
                    mass displacement curves based on the stage of the U-turn as well
                    as whether or not the filament performed an upward or downward flip. 

INPUT
ARGUMENT(S):        N/A
                

CREATED:            20Jun23

MODIFICATIONS
LOG:                N/A
    
            
LAST MODIFIED
BY:                 Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:             3.9.13

VERSION:            1.0

AUTHOR(S):          Thomas Nguyen

STATUS:             Working

TO DO LIST:         N/A

NOTE(S):        
    
1)                  N/A

"""
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def ANOVA2_netycom_phase_flipdir(input_file,output_directory,file_name):
    """
    This function performs a 2-way ANOVA to see if there's a significant relationship
    between net displacement due to drift, phase of the transition, and 
    Displacement Type. After performing the ANOVA, it will perform an ad-hoc 
    Tukey test as an FDR to see which relationships are significant.
    
    Inputs:
    
    input_file:                 Input Pandas dataframe that lists the measured
                                data values for each U-turn flipping event.
    output_directory:           Path to directory to be created.
    file_name:                  The file name to be used for the output graphical
                                files.
    """
    
    ### Perform 2-way ANOVA ###
    input_file['Stage_Displacement_Type'] = input_file['Stage'].astype(str) +\
        '|' + input_file['Displacement Type']
    model = smf.ols("Q('Stage ABS Adjusted Center of Mass-y') ~ Q('Stage') + Q('Displacement Type') + Q('Stage') :Q('Displacement Type')",
                    data = input_file).fit()
    sm.stats.anova_lm(model, typ=2)
    
    ### Perform Ad hoc corrections for 2-way ANOVA ###
    m_comp = pairwise_tukeyhsd(endog=input_file['Stage ABS Adjusted Center of Mass-y'],
                               groups=input_file['Stage_Displacement_Type'], alpha=0.05)
    tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:],
                              columns = m_comp._results_table.data[0])
    tukey_data['Stage_1'] = tukey_data['group1'].apply(lambda x: x.split('|')[0])
    tukey_data['Displacement_Type_1'] = tukey_data['group1'].apply(lambda x: x.split('|')[1])
    tukey_data['Stage_2'] = tukey_data['group2'].apply(lambda x: x.split('|')[0])
    tukey_data['Displacement_Type_2'] = tukey_data['group2'].apply(lambda x: x.split('|')[1])
    tukey_data = tukey_data[((tukey_data['Stage_1'] == tukey_data['Stage_2']) &\
                             (tukey_data['Displacement_Type_1'] !=\
                              tukey_data['Displacement_Type_2'])) |\
                            ((tukey_data['Stage_1'] != tukey_data['Stage_2']) &\
                                                     (tukey_data['Displacement_Type_1'] ==\
                                                      tukey_data['Displacement_Type_2']))]
    
    new_file_name = '{}.csv'.format(file_name)
    tukey_data.to_csv(os.path.join(output_directory,new_file_name))
    return tukey_data