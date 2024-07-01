# -*- coding: utf-8 -*-
import os, re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

project_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/02_Shear_Migration/00_Scripts/01b_Non_Brownian_Analysis/'
os.chdir(project_dir)

from calculations.center_of_mass import center_of_mass
from calculations.first_derivative import first_derivative
from calculations.second_derivative import second_derivative
from calculations.adjust_position_data import adjust_position_data

main_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/02_Shear_Migration/00_Scripts/01_Migration_Simulations/02_Actual_Results/NB_Poiseuille_J/'
redo_dir = 'C:/Users/super/OneDrive - University of California, Davis/Research/00_Projects/02_Shear_Migration/00_Scripts/01_Migration_Simulations/02_Actual_Results/NB_Poiseuille_J_redo/'

check_dir = 'UC_1p00_MB_5p00e4/Down_0p25/'
main_dir = os.path.join(main_dir,check_dir)
redo_dir = os.path.join(redo_dir,check_dir)

#%%

redo_position_data = np.load(os.path.join(
    redo_dir,'filament_allstate.npy'))
redo_tension_data = np.load(os.path.join(
    redo_dir,'filament_tension.npy'))
redo_params_csv = pd.read_csv(os.path.join(
    redo_dir,'parameter_values.csv'),index_col = 0,header = 0)
redo_time_data = np.load(os.path.join(
    redo_dir,'filament_time_vals_all.npy'))

main_position_data = np.load(os.path.join(
    main_dir,'filament_allstate.npy'))
main_tension_data = np.load(os.path.join(
    main_dir,'filament_tension.npy'))
main_params_csv = pd.read_csv(os.path.join(
    main_dir,'parameter_values.csv'),index_col = 0,header = 0)
main_time_data = np.load(os.path.join(
    main_dir,'filament_time_vals_all.npy'))


print((redo_position_data[:,1,:] - main_position_data[:,1,:]).min(),
(redo_position_data[:,1,:] - main_position_data[:,1,:]).max())
#%%
redo_center_mass = center_of_mass(position_data = redo_position_data,
                                  position= 0,
                                  dim = 3,
                                  adj_centering = None,
                                  adj_translation = True,
                                  transl_val = 0.25)
main_center_mass = center_of_mass(position_data = main_position_data,
                                  position= 0,
                                  dim = 3,
                                  adj_centering = None,
                                  adj_translation = True,
                                  transl_val = 0.25)

print((redo_center_mass[:,1] - main_center_mass[:,1]).min(),
(redo_center_mass[:,1] - main_center_mass[:,1]).max())

#%%
print((redo_tension_data[:,1] - main_tension_data[:,1]).min(),
(redo_tension_data[:,1] - main_tension_data[:,1]).max())

fig,axes = plt.subplots()
axes.plot(np.linspace(-1/2,1/2,101),redo_tension_data[:,1],'r')
axes.plot(np.linspace(-1/2,1/2,101),main_tension_data[:,1],'b')
plt.show()

#%%
fig,axes = plt.subplots()
axes.plot(redo_position_data[:,0,1],redo_position_data[:,1,1],'r')
axes.plot(main_position_data[:,0,1],main_position_data[:,1,1],'b')
axes.set_ylim(-0.6,0.6)
axes.set_xlim(-0.6,0.6)
# plt.axis('square')
plt.show()


#%%
data_diff_lst = []

for root,dirs,files in os.walk(redo_dir):
    for subdir_ in dirs:
        path_to_dir = os.path.join(root,subdir_)
        check_file = os.path.join(path_to_dir,'filament_allstate.npy')
        check_file2 = os.path.join(path_to_dir,'parameter_values.csv')
        if os.path.exists(check_file) and os.path.exists(check_file2):
        
            redo_position_data = np.load(os.path.join(
                path_to_dir,'filament_allstate.npy'))
            redo_tension_data = np.load(os.path.join(
                path_to_dir,'filament_tension.npy'))
            redo_params_csv = pd.read_csv(os.path.join(
                path_to_dir,'parameter_values.csv'),index_col = 0,header = 0)
            redo_time_data = np.load(os.path.join(
                path_to_dir,'filament_time_vals_all.npy'))
            
            path_split = path_to_dir.split('/')
            main_dir_equiv = os.path.join(main_dir,path_split[-1])
            main_position_data = np.load(os.path.join(
                main_dir_equiv,'filament_allstate.npy'))
            main_tension_data = np.load(os.path.join(
                main_dir_equiv,'filament_tension.npy'))
            main_params_csv = pd.read_csv(os.path.join(
                main_dir_equiv,'parameter_values.csv'),index_col = 0,header = 0)
            main_time_data = np.load(os.path.join(
                main_dir_equiv,'filament_time_vals_all.npy'))
            
            
            mu_bar = int(redo_params_csv.loc['Mu_bar','Value'])
            channel_h = float(redo_params_csv.loc['Channel Upper Height','Value'])
            U_centerline = float(redo_params_csv.loc['Poiseuille U Centerline','Value'])
            vert_displ = float(redo_params_csv.loc['Vertical Displacement','Value'])
            
            redo_center_mass = center_of_mass(position_data = redo_position_data,
                                              position= 0,
                                              dim = 3,
                                              adj_centering = None,
                                              adj_translation = True,
                                              transl_val = vert_displ)
            main_center_mass = center_of_mass(position_data = main_position_data,
                                              position= 0,
                                              dim = 3,
                                              adj_centering = None,
                                              adj_translation = True,
                                              transl_val = vert_displ)
            
            diff_dict = {"Time": redo_time_data,
                         "Mu_bar":mu_bar,
                         "Channel Height": channel_h,
                         "Poiseuille U Centerline": U_centerline,
                         "Starting Vertical Displacement": vert_displ,                             
                         "Max Difference in Y-Position Data": (
                             redo_position_data[:,1,:] - main_position_data[:,1,:]).max(axis = 0),
                         "Min Difference in Y-Position Data": (
                             redo_position_data[:,1,:] - main_position_data[:,1,:]).min(axis = 0),
                         "Max Difference in Y-Center of Mass Data": (
                             redo_center_mass[:,1] - main_center_mass[:,1]).max(axis = 0),
                         "Min Difference in Y-Center of Mass Data": (
                             redo_center_mass[:,1] - main_center_mass[:,1]).min(axis = 0),
                         "Max Difference in Tension Data":
                             (redo_tension_data - main_tension_data).max(axis = 0),
                         "Min Difference in Tension Data":
                             (redo_tension_data - main_tension_data).min(axis = 0)
                         }
            data_diff_lst.append(diff_dict)

data_diff_df = pd.concat(
    [pd.DataFrame.from_dict(i) for i in data_diff_lst],ignore_index = True)               
#%%
exp_groups = data_diff_df.groupby(by = ['Mu_bar','Channel Height',
                                        'Poiseuille U Centerline','Starting Vertical Displacement'])
new_diff_list = []
for group in exp_groups.groups.keys():
    group_df = exp_groups.get_group(group)
    mu_bar,channel_h,U_centerline,vert_displ = group
    
#%%
fil_df = data_diff_df[data_diff_df['Mu_bar'] == 5e4]
fig,axes = plt.subplots()
sns.lineplot(x = 'Time',
                y = 'Max Difference in Y-Position Data',
                hue = 'Starting Vertical Displacement',
                data= fil_df)

plt.show()
            
            
            