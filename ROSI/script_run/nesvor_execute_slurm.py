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

    MARSFET_MESO_tru_INIT = "/home/cmercier/results/tru"
    
    MARSFET_DATABASE = "/scratch/cmercier/code/pyrecon/bd_clinique.csv"

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
            #if  subject == "sub-0148" and session == "ses-0174":
            #    print("this subject exist hehehe")
            if subject == "sub-0379"  or subject == "sub-0567" or subject == "sub-0703" :
            #if subject in sub_list and session in ses_list :
            #
            #
            #
            #subject in sub_list and session in ses_list :
            #
            # 
            #
            #subject in sub_list and session in ses_list :
            #
            #== "sub-0009" and session == "ses-0012":
            #
                joblib_path = os.path.join(job_res,subject, session,'res.joblib.gz')
                print(joblib_path)
                input_slices = os.path.join(output,'svort',subject,session,'res')
                #print(input_sl)
                #print("input_stacks",input_stacks)
            
                
                output_tru = os.path.join(output,'nesvor', 'rosi', subject, session)
                output_tru_mask = os.path.join(output,'nesvor', 'rosi_mask', subject, session)
                output_tru_similarity = os.path.join('/data','nesvor', 'rosi', subject, session)
                output_nesvor = os.path.join('/data','nesvor',subject,session,"volume_rosi_svort2.nii")
                path_to_mask = os.path.join(output,'nesvor',subject,session,"volume_mask.nii")
                path_to_volume = os.path.join(output,'nesvor',subject,session,"volume_rosi_svort2.nii")
               
                #if True :
                print("sub :",subject," ses :",session)
                print(os.path.exists(joblib_path),not os.path.exists(path_to_volume)) 
                if True:
                #os.path.exists(joblib_path) and not os.path.exists(path_to_volume):
                    cmd_os_1 = " --input_slices " + input_slices
                    cmd_os_1 += " --output " + output_tru
                    cmd_os_1 += " --output_mask " + output_tru_mask
                    cmd_os_1 += " --results " + joblib_path
                    cmd_os_2 =  " --input-slices " + output_tru_similarity
                    cmd_os_2 += " --output-volume " + output_nesvor
                    cmd_os_2 += " --registration none "
                    cmd_os_2 += " --no-transformation-optimization "
                    cmd_os_2 += " --output-resolution 0.5"
                    cmd_os_2 += " --inference-batch-size 255 "
                    cmd_os_2 += " --n-inference-samples 128 "
                    #cmd_os_2 += " --inference-batch-size 32 "
                    #cmd_os_2 += " --n-inference-samples 128 "
                    #cmd_os_2 += " --single-precision "
                    

                    cmd = (
                        "sbatch"
                        + " "
                        + "/scratch/cmercier/code/pyrecon/ROSI/utils/slurm/rosi_nesvor.slurm"
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



                    

