# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 21:13:38 2023

@author: super
"""

import os, re, argparse, shutil, sys
import pandas as pd


from create_dir import create_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_directory", 
                        help="The main directory where the simulation files currently reside in",
                    type = str)
    # args = parser.parse_args(['C:/Users/super/OneDrive - University of California, Davis/School/UCD_Files/Work/00_Projects/02_Shear_Migration/00_Scripts/01_Migration_Simulations/01_Test_Results/Poiseuille_Flow/'])
    args = parser.parse_args()
    file_rename_change_lst = []
    flow_type = 'Poiseuille'
    for root, dirs, files in os.walk(args.input_directory):
        for file in files:
            if file != 'file_name_changes.csv':
                file_path = os.path.join(root,file)
                
                ### Break file path down based on file name and sub-directories ###
                path_components = [i for i in reversed(os.path.normpath(file_path).lstrip(os.path.sep).split(os.path.sep))]
                path_to_delete = [i for i in reversed(path_components.copy()[1:])] ### Keep sub-directories and not file name ###
                
                ### Check if this is a failed log file or not ###
                file_name,above_path = path_components[:2]
                failed_file_name = re.search(r"failed_run_log_(\d{2}).csv",file_name)
                if failed_file_name:
                    if above_path == 'failed_logs':
                        rep_dir,mb_dir,run_dir,vd_dir = path_components[2:6]
                        file_name = failed_file_name
                        rest_of_path = path_components[6:]
                else:
                    file_name,rep_dir,mb_dir,run_dir,vd_dir = path_components[:5]
                    rest_of_path = path_components[5:]
                
                ### Determine ensemble number ###
                rep_match = re.search(r"rep_(\d{1,})",rep_dir)
                if not rep_match:
                    rep_match2 = re.search(r"R_(\d{1,})",rep_dir)
                    if rep_match2:
                        rep_num = rep_match2.group(1)
                else:
                    rep_num = rep_match.group(1)
                
                ### Determine Mu_bar value ###
                mb_match = re.search(r"Mu_bar_(\d{1,})",mb_dir)
                if not mb_match:
                    mb_match2 = re.search(r"MB_(\d{1,})",mb_dir)
                    if mb_match2:
                        mb_num = mb_match2.group(1)
                else:
                    mb_num = mb_match.group(1)
                
                ### Determine run conditions ###
                if flow_type == 'Poiseuille':
                    run_match = re.search(r"(.*)_U_(.*)",run_dir)
                    if not run_match:
                        run_match2 = re.search(r"(.*)_UC_(.*)",run_dir)
                        if run_match2:
                            rigidity_profile,U_centerline_text = run_match2.group(1),run_match2.group(2)
                    else:
                        rigidity_profile,U_centerline_text = run_match.group(1),run_match.group(2)
                elif flow_type == 'Shear':
                    run_match = re.search(r"(.*)_H_(.*)",run_dir)
                    if run_match:
                        rigidity_profile,channel_height = run_match.group(1),run_match.group(2)
                        
                ### Determine Vertical Displacement ###
                vd_match = re.search(r"Vert_Displ_(.*)",vd_dir)
                if not vd_match:
                    vd_match2 = re.search(r"VD_(.*)",vd_dir)
                    if vd_match2:
                        vd_val = vd_match2.group(1)
                else:
                    vd_val = vd_match.group(1)
                
                ### Assemble new path ###
                parent_path = [i for i in reversed(rest_of_path)]
                parent_path = os.path.join(*parent_path)
                
                if not failed_file_name:
                    if flow_type == 'Poiseuille':
                        new_file_path = ["VD_{}".format(vd_val),"{}_UC_{}".format(rigidity_profile,U_centerline_text),
                                         "MB_{}".format(mb_num),"R_{}".format(rep_num),file_name]
                    elif flow_type == 'Shear':
                        new_file_path = ["VD_{}".format(vd_val),"{}_H_{}".format(rigidity_profile,channel_height),
                                         "MB_{}".format(mb_num),"R_{}".format(rep_num),file_name]
                    new_file_path = os.path.join(*new_file_path)
                    new_file_path = os.path.join(parent_path,new_file_path)
                    new_dir_path = os.path.split(new_file_path)[0]
                    
                    create_dir(new_dir_path)

                else:
                    if flow_type == 'Poiseuille':
                        new_file_path = ["VD_{}".format(vd_val),"{}_UC_{}".format(rigidity_profile,U_centerline_text),
                                         "MB_{}".format(mb_num),"R_{}".format(rep_num),above_path,file_name]
                    elif flow_type == 'Shear':
                        new_file_path = ["VD_{}".format(vd_val),"{}_H_{}".format(rigidity_profile,channel_height),
                                         "MB_{}".format(mb_num),"R_{}".format(rep_num),above_path,file_name]
                        
                    new_file_path = os.path.join(*new_file_path)
                    new_file_path = os.path.join(parent_path,new_file_path)
                    new_dir_path = os.path.split(os.path.split(new_file_path)[0])[0]
                    
                    create_dir(new_dir_path)
                path_to_rm = os.path.join(*path_to_delete)
                
                
                ### Move directories and delete old ones ###
                try:
                    os.replace(file_path,new_file_path)
                except FileNotFoundError:
                    print(file_path,new_file_path)
                    # pass
                
                
                ### Append change to new dictionary and eventually pandas dataframe ###
                # file_change = {"Old File Path": [file_path],
                #                "Old Path to Delete": path_to_rm,
                #                "New File Path": new_file_path}
                # file_rename_change_lst.append(file_change)
            
                         
    # file_rename_dict = pd.concat(
    #     [pd.DataFrame.from_dict(i) for i in file_rename_change_lst],ignore_index = True)
    # file_rename_dict.to_csv(os.path.join(args.input_directory,'file_name_changes.csv'))
    
    sys.exit(1)
    

    