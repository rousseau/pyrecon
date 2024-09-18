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

    MARSFET_MESO_RESULTS = "/home/cmercier/results/"

    MARSFET_MESO_SVORT_INIT = "/home/cmercier/results/svort"
    
    MARSFET_DATABASE = "/scratch/cmercier/code/pyrecon/bd_chapter4.csv"

    job_res = MARSFET_MESO_ROSI
    output = MARSFET_MESO_RESULTS
    slices_path = MARSFET_MESO_SVORT_INIT
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
            if subject == "sub-0002" and session == "ses-0002":
            #subject in sub_list and session in ses_list : 
            #
            #
            #
            #
            #

                input_stacks = os.listdir(os.path.join(stacks_path,subject, session))
                list_stacks=[]
                list_masks=[]
                for file in input_stacks:
                    if file.endswith("denoised_T2w.nii.gz") and 'haste' in file:
						#stack = os.path.join(dir_reconst, subject+ "_"+ session + "_"+ "acq-"+ sequence+ "_"+ "run" + "-" + serie + "_desc-denoised_T2w.nii.gz")
                        path_to_file = os.path.join(stacks_path,subject,session,file)
                        list_stacks.append(path_to_file)
                        run = file.find("run") #find the number of the run to make sure the stack is associated with its corresponding mask
                        num_index = run + 4
                        num = "run-%s" %(file[num_index])
                        print("number run",num)
                        for file in input_stacks:
                            if file.endswith("brainmask_T2w.nii.gz") and 'haste' in file and num in file:
                                path_to_file = os.path.join(stacks_path,subject,session,file)
                                list_masks.append(path_to_file)
                                break
                print(list_stacks)
                print(list_masks)
                list_stacks = ' '.join(str(list_stacks) for list_stacks in list_stacks)
                print(list_stacks)
                list_masks = ' '.join(str(list_masks) for list_masks in list_masks)
                #dir_out = os.path.join(output_data, subject, session,'res_alone')
                #print(dir_out)
                #subject in sub_list and session in ses_list :
                #
                #subject in sub_list and session in ses_list :
                #
                # 
                #
                #subject in sub_list and session in ses_list :
                #
                #== "sub-0009" and session == "ses-0012":
                #== "sub-0002" and session == "ses-0002":
                joblib_path = os.path.join(job_res,subject, session, 'res_alone/res_mse.joblib.gz')
                input_slices = os.path.join(slices_path,subject,session,'res')
                #print(input_sl)
                #print("input_stacks",input_stacks)
            
                
                output_svort = os.path.join(output,'nesvor', 'rosi_ms', subject, session)
                output_svort_mask = os.path.join(output,'nesvor', 'rosi_ms_mask', subject, session)
                output_svort_similarity = os.path.join('/data','nesvor', 'rosi_ms', subject, session)
                output_nesvor = os.path.join('/data','nesvor',subject,session,"volume_rosi_ms.nii")
                output_slices = os.path.join('/data','nesvor',subject,session,'slices_opti')
                path_to_mask = os.path.join(output,'nesvor',subject,session,"volume_mask.nii")
                path_to_volume = os.path.join(output,'nesvor',subject,session,"volume_rosi_ms.nii")
               
                if True : 
                #os.path.exists(joblib_path) and not os.path.exists(path_to_volume):
                    print(list_stacks)
                    cmd_os_1 = " --input_stacks " + list_stacks
                    cmd_os_1 += " --input_mask " + list_masks
                    cmd_os_1 += " --results " + joblib_path
                    cmd_os_1 += " --output " + output_svort

                    cmd_os_2 =  " --input-slices " + output_svort_similarity
                    cmd_os_2 += " --output-volume " + output_nesvor
                    #cmd_os_2 += " --output-slices " +  output_slices
                    cmd_os_2 += " --registration none "
                    cmd_os_2 += " --no-transformation-optimization "
                    cmd_os_2 += " --output-resolution 0.5"
                    cmd_os_2 += " --inference-batch-size 255 "
                    cmd_os_2 += " --n-inference-samples 128 "
                    cmd_os_2 += " --single-precision "
                    

                    cmd = (
                        "sbatch"
                        + " "
                        + "/scratch/cmercier/code/pyrecon/ROSI/utils/slurm/rosi_nesvor_alone.slurm"
                        + " "
                        + '"'
                        + cmd_os_1
                        + '"'
                        + " "
                        + '"'
                        + cmd_os_2
                        + '"'
                        + " "
                        + MARSFET_MESO_RESULTS
                        )
                    
                    os.system(cmd)



                    

