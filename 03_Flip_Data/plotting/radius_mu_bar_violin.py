"""
FILE NAME:              radius_mu_bar_violin.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):                Process_Filament_Flip_Data.py

DESCRIPTION:            This script contains a function that will plot the caculated radius
                        of bending of the filament flipping events as  a function of mu_bar 
                        on a violin plot.
                       

INPUT
FILES(S):               N/A

OUTPUT
FILES(S):               1) .PNG/.PDF/.EPS file that shows the violinplot relationship between
                        radius and mu_bar.


INPUT
ARGUMENT(S):            N/A

CREATED:        22Nov22

MODIFICATIONS
LOG:
22Nov22                 1) Migrated code to generate the plots from the original script
                        to its own instance here.
26Sep23                 2) Separated net drift and radius functions into separate scripts. 

    
            
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

NOTE(S):                This script requires the 'statsannotations' library. With the current
                        version (v 0.5) requires an older version of seaborn (v <= 0.12).

"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statannotations.Annotator import Annotator


### Use LaTeX to generate plots ###
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})


def radius_mu_bar_violin(input_file,p_vals_df,flow_type,output_directory,file_name):
    """
    This function will take in the filament flipping data and plot the calculated radius 
    of bending based on the different mu_bar values and flipping direction
    on a violin plot.
    
    Inputs:
    
    input_file:                 Input Pandas dataframe that lists the measured
                                data values for each U-turn flipping event.
    p_vals_df:                  Pandas DataFrame that contains the FDR-adjusted p-values
                                corresponding to each comparison.
    flow_type:                  String to denote the background flow type.
    output_directory:           Path to directory to be created.
    file_name:                  The file name to be used for the output graphical
                                files.
    """
    
    if flow_type == 'Poiseuille':
        hue_order_fix = [r'$5 \times 10^{4}$',r'$1 \times 10^{5}$',
                 r'$2 \times 10^{5}$',r'$5 \times 10^{5}$']
    elif flow_type == 'Shear':
        hue_order_fix = [r'$5 \times 10^{4}$',r'$1 \times 10^{5}$',
                 r'$2 \times 10^{5}$']
    p_vals_df.sort_values(by='Mu_bar_1', key=lambda x: x.map({val: i for i, val in enumerate(hue_order_fix)}),
                          inplace = True)

    cblind_palette = ["#A9E5BB","#FCF6B1"]
        
    fig,axes = plt.subplots(figsize = (10,7))
    g = sns.violinplot(x = 'Mu_bar String',y = 'Radius of Bending-1',
                               data = input_file,hue = 'Displacement Type',
                               palette = cblind_palette,
                               order = hue_order_fix,
                               inner = 'quartile',linewidth = 2,width = 0.9)

    median_lines = g.lines[1::3]
    other_lines = [i for i in g.lines if i not in median_lines]
    for l in other_lines:
        l.remove()
    for l in median_lines:
        l.set_linestyle('dashed')
        l.set_linewidth(2)
        l.set_color('black')
        l.set_alpha(1)
    axes.tick_params(axis='both', which='both', direction = 'in',labelsize=11)
    if flow_type == 'Shear':
        axes.set_ylim(0.055,0.105)
        axes.set_yticks(np.linspace(0.06,0.10,5))
    elif flow_type == 'Poiseuille':
        axes.set_ylim(0.03,0.105)
        axes.set_yticks(np.linspace(0.04,0.10,4))
    axes.tick_params(axis='both', which='both', direction = 'in',labelsize=15)
    axes.set_xlabel(r"$\bar{\mu}$",fontsize = 17,labelpad = 5)
    axes.set_ylabel(r"$R$",fontsize = 17,labelpad = 5)
    axes.legend(loc='upper right', 
                prop={'size': 15},title= "Displacement Type").get_title().set_fontsize("17")
    axes.set_aspect((np.diff(axes.get_xlim()))/(1.25*np.diff(axes.get_ylim())))
    
    ### Draw Lines to Show Significant p-values ###
    pairs = [((i[-4],i[-3]),(i[-2],i[-1])) for i in p_vals_df.itertuples()]
    pvals = [i[4] for i in p_vals_df.itertuples()]
    annotator = Annotator(
    axes, pairs, data=input_file, x = 'Mu_bar String',y = 'ABS Net COM-y',
    hue = 'Displacement Type',order = hue_order_fix)
    annotator.configure(text_format="star", loc="inside",fontsize = 'x-large')
    annotator.set_pvalues(pvals)
    if flow_type == 'Shear':
        annotator.annotate(line_offset_to_group = 0.3)
    elif flow_type == 'Poiseuille':
        annotator.annotate(line_offset_to_group = 0.4)
            
    fig.savefig(os.path.join(output_directory,'{}.png'.format(file_name)),
                bbox_inches = 'tight',dpi = 600)
    fig.savefig(os.path.join(output_directory,'{}.pdf'.format(file_name)),
                format = 'pdf',bbox_inches = 'tight',dpi = 600)
    fig.savefig(os.path.join(output_directory,'{}.eps'.format(file_name)),
                bbox_inches = 'tight',format = 'eps',dpi = 600)
    plt.show()
    