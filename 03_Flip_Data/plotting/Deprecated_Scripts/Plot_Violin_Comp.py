# -*- coding: utf-8 -*-
"""
FILE NAME:      B__v00_00_Plot_Net_Displacement_Mu_bar.py

COMPLEMENTARY
SCRIPT(S)/
FILE(S):        A__v01_03_Process_Filament_Flip_Data.py

DESCRIPTION:    This script will plot the net displacement of the filament flipping
                events as a function of mu_bar on a violin plot.

INPUT
FILES(S):       1) .CSV file that contains the radius of bending, net 
                displacement, and other parameters based on instances of flipping.

OUTPUT
FILES(S):       1) .PNG file that shows the violinplot relationship between
                net displacement and mu_bar.
                2) .PDF file that shows the violinplot relationship between
                net displacement and mu_bar.
                1) .EPS file that shows the violinplot relationship between
                net displacement and mu_bar.


INPUT
ARGUMENT(S):    1) Input Data File: The true path to the master .CSV file
                that contains the all information regarding U-turns in flow
                where the filament is presumed to make a J-shape. 
                2) Main Output directory: The directory that will house all of the
                output files associated with the analysis; if it doesn't exist,
                it will be created.
                3) File Name: The name the graphical files will be saved as.

CREATED:        22Nov22

MODIFICATIONS
LOG:
22Nov22         1) Migrated code to generate the plots from the original script
                to its own instance here.

    
            
LAST MODIFIED
BY:             Thomas Nguyen

PYTHON
VERSION USED
TO WRITE
SCRIPT:         3.8.8

VERSION:        1.0

AUTHOR(S):      Thomas Nguyen

STATUS:         Working

TO DO LIST:     N/A

NOTE(S):        N/A

"""
import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statannotations.Annotator import Annotator


### Use LaTeX to generate plots ###
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Times",
    'text.latex.preamble': r'\usepackage{amsmath}'})

#%%

def plot_net_displ_mu_bar_violin(input_file,p_vals_df,flow_type,output_directory,file_name):
    """
    This function will take in the filament flipping data and plot the net 
    displacement as as function of mu_bar on a violinplot.
    
    Inputs:
    
    input_file:                 Input Pandas dataframe that lists the measured
                                data values for each U-turn flipping event.
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
    g = sns.violinplot(x = 'Mu_bar String',y = 'ABS Net COM-y',
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
    axes.tick_params(axis='both', which='both', direction = 'in',labelsize=15)
    axes.set_ylim(-0.01,0.09)
    axes.set_yticks(np.linspace(0,0.08,5))
    axes.set_xlabel(r"$\bar{\mu}$",fontsize = 17,labelpad = 5)
    axes.set_ylabel(r"$|\Delta y^{\text{com}}|$",fontsize = 17,labelpad = 5)
    axes.legend(loc='upper right', 
                prop={'size': 15},title= "Displacement ").get_title().set_fontsize("17")
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
        annotator.annotate(line_offset_to_group = 0.35)
            
    fig.savefig(os.path.join(output_directory,'{}.png'.format(file_name)),
                bbox_inches = 'tight',dpi = 400)
    fig.savefig(os.path.join(output_directory,'{}.pdf'.format(file_name)),
                format = 'pdf',bbox_inches = 'tight',dpi = 400)
    fig.savefig(os.path.join(output_directory,'{}.eps'.format(file_name)),
                bbox_inches = 'tight',format = 'eps',dpi = 400)
    plt.show()
    
def plot_radius_mu_bar_violin(input_file,p_vals_df,flow_type,output_directory,file_name):
    """
    This function will take in the filament flipping data and plot the net 
    displacement as as function of mu_bar on a violinplot.
    
    Inputs:
    
    input_file:                 Input Pandas dataframe that lists the measured
                                data values for each U-turn flipping event.
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
                bbox_inches = 'tight',dpi = 400)
    fig.savefig(os.path.join(output_directory,'{}.pdf'.format(file_name)),
                format = 'pdf',bbox_inches = 'tight',dpi = 400)
    fig.savefig(os.path.join(output_directory,'{}.eps'.format(file_name)),
                bbox_inches = 'tight',format = 'eps',dpi = 400)
    plt.show()
    

def violinplot_statistics_net_displ(input_file,output_directory,file_name):
    """
    This function performs a 2-way ANOVA to see if there's a significant relationship
    abetween net displacement due to drift, mu_bar, and displacement type. After
    performing the ANOVA, it will perform an ad-hoc Tukey test as an FDR to see
    which relationships are significant.
    
    Inputs:
        
    Inputs:
    
    input_file:                 Input Pandas dataframe that lists the measured
                                data values for each U-turn flipping event.
    output_directory:           Path to directory to be created.
    file_name:                  The file name to be used for the output graphical
                                files.
    """
    
    ### Perform 2-way ANOVA ###
    
    input_file['Mu_bar_Displacement_Type'] = input_file['Mu_bar String'].astype(str) + '|' + input_file['Displacement Type']
    model = smf.ols("Q('ABS Net COM-y') ~ Q('Displacement Type') + Q('Mu_bar String') + Q('Displacement Type') :Q('Mu_bar String')",data = input_file).fit()
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

def violinplot_statistics_radius(input_file,output_directory,file_name):
    """
    This function performs a 2-way ANOVA to see if there's a significant relationship
    abetween net displacement due to drift, mu_bar, and displacement type. After
    performing the ANOVA, it will perform an ad-hoc Tukey test as an FDR to see
    which relationships are significant.
    
    Inputs:
        
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
#%%

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project_directory",
                        help="Specify the path to the directory that contains this script and all other relevant scripts",
                    type = str)
    parser.add_argument("input_data_file",
                        help="Specify the Absolute Path to the CSV file that contains the filament flip data",
                    type = str,default = None)
    parser.add_argument("output_directory",
                        help = "Specify the directory where the resulting plots will be saved in",
                        type = str,default = None)
    parser.add_argument("file_name",
                        help = "Specify the file name for the graphs that will be saved",
                        type = str,default = None)
    # args = parser.parse_args() #Uncomment this line if running from a separate console
    args = parser.parse_args(['C://Users//super//OneDrive - University of California, Davis//School//UCD_Files//Work//00_Projects//02_Shear_Migration//00_Scripts//03_Flip_Data//',
                              'C://Users//super//OneDrive - University of California, Davis//School//UCD_Files//Work//00_Projects//02_Shear_Migration//00_Scripts//03_Flip_Data//02_Actual_Results//Poiseuille_Flow_Walls//Poiseuille_H_0p50_Data_Flip_Filtered.csv',
                              'C://Users//super//OneDrive - University of California, Davis//School//UCD_Files//Work//00_Projects//02_Shear_Migration//00_Scripts//03_Flip_Data//02_Actual_Results//Poiseuille_Flow_Walls//violin_plots_displacement//',
                              'Poiseuille_walls_H_0p50'])
    
    os.chdir(args.project_directory)
    from misc.create_dir import create_dir
    
    ensemble_data_df = pd.read_csv(args.input_data_file,index_col = 0,header = 0)
    ensemble_data_df['Mu_bar'] = ensemble_data_df['Mu_bar'].astype(int)
    create_dir(args.output_directory)
    
    ### Net Displacement vs. Mu_bar ###
    net_displ_tukey_df = violinplot_statistics_net_displ(input_file = ensemble_data_df,
                                     output_directory= args.output_directory,
                                     file_name = 'poiseuille_walls_net_displ_stat')
    
    plot_net_displ_mu_bar_violin(input_file = ensemble_data_df,
                                     output_directory= args.output_directory,
                                     p_vals_df = violinplot_statistics_net_displ,
                                     file_name = '{}_{}'.format(args.file_name,'net_displ'))
    
    ### Radius of Bending-1 vs. Mu_bar ###
    radius_tukey_df = violinplot_statistics_net_displ(input_file = ensemble_data_df,
                                     output_directory= args.output_directory,
                                     file_name = 'poiseuille_walls_radius_stat')
    
    plot_radius_mu_bar_violin(input_file = ensemble_data_df,
                                     output_directory= args.output_directory,
                                     p_vals_df = violinplot_statistics_net_displ,
                                     file_name = '{}_{}'.format(args.file_name,'radius'))