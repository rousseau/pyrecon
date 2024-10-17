import os
import re
import pathlib
import subprocess
import csv
import nisnap
import nibabel as nib
import numpy as np

if __name__ == "__main__":


    MARSFET_DATAPATH = "/scratch/gauzias/data/datasets/MarsFet/derivatives/preprocessing"  

    MARSFET_MESO_ROSI = "/scratch/cmercier/results/rosi/"

    MARSFET_MESO_RESULTS = "/scratch/cmercier/results/"

    MARSFET_MESO_tru_INIT = "/home/cmercier/results/tru"
    
    MARSFET_DATABASE = "/scratch/cmercier/code/pyrecon/bd_chapter4.csv"

    job_res = MARSFET_MESO_ROSI
    output = MARSFET_MESO_RESULTS
    slices_path = MARSFET_MESO_tru_INIT
    csv_file =  MARSFET_DATABASE
    stacks_path = MARSFET_DATAPATH
    sub_list = []
    ses_list = []
    with open(csv_file,newline='') as csvfile:
        readrow = csv.reader(csvfile,delimiter=',')
        for row in readrow:
            sub_list.append(row[0])
            ses_list.append(row[1])

    subjects = [sub for sub in os.listdir(stacks_path) if os.path.isdir(os.path.join(stacks_path,sub))]
    subjects.sort()

    for subject in subjects:
        dir_subject = os.path.join(stacks_path, subject)
        sessions = os.listdir(dir_subject)
        for session in sessions:
            #if  :
            if  subject == "sub-0002" and session == "ses-0002":
            #    print("this subject exist hehehe")
                input_stacks = os.listdir(os.path.join(stacks_path,subject, session))
                list_stacks=[]
                list_masks=[]
                for file in input_stacks:
                    if file.endswith("denoised_T2w.nii.gz") and 'tru' in file:
						#stack = os.path.join(dir_reconst, subject+ "_"+ session + "_"+ "acq-"+ sequence+ "_"+ "run" + "-" + serie + "_desc-denoised_T2w.nii.gz")
                        path_to_file = os.path.join('/data',subject,session,file)
                        list_stacks.append(path_to_file)
                        run = file.find("run") #find the number of the run to make sure the stack is associated with its corresponding mask
                        desc = file.find("_desc")
                        num_index_min = run + 4
                        num_index_max = desc
                        num = "run-%s_" %(file[num_index_min:num_index_max])
                        for file in input_stacks:
                            if file.endswith("brainmask_T2w.nii.gz") and 'tru' in file and num in file:
                                path_to_file = os.path.join('/data',subject,session,file)
                                list_masks.append(path_to_file)
                                break
                print(list_stacks)
                print(list_masks)
                list_stacks = ' '.join(str(list_stacks) for list_stacks in list_stacks)
                print(list_stacks)
                list_masks = ' '.join(str(list_masks) for list_masks in list_masks)
              
    
                dir_output = os.path.join('/results','niftymic', 'rosi', subject, session)
                dir_input = os.path.join('/results','rosi',subject,session,"res_alone/niftymic_mvt")
                path_to_reconstruction = "/scratch/cmercier/code/pyrecon/ROSI/rosi/reconstruction"
                path_to_res ="/scratch/cmercier/results/"
                
               
                if True :
                    cmd_os = "python /app/NiftyMIC/niftymic_run_reconstruction_pipeline_slices.py"
                    cmd_os += " --filenames " + list_stacks
                    cmd_os += " --filenames-masks " + list_masks
                    cmd_os += " --dir-output " + dir_output
                    cmd_os += " --dir-input-mc " + dir_input             

                    cmd = (
                        "sbatch"
                        + " "
                        + "/scratch/cmercier/code/pyrecon/ROSI/utils/slurm/rosi_niftymic.slurm"
                        + " "
                        + '"'
                        + cmd_os
                        + '"'
                        + " "
                        + MARSFET_DATAPATH
                        + " "
                        + path_to_reconstruction
                        + " "
                        + path_to_res
                        + " "
                        )
                    
                    os.system(cmd)



                    

