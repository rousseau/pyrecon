import os
import re
import pathlib
import subprocess
import csv


if __name__ == "__main__":


    MARSFET_DATAPATH = "/scratch/gauzias/data/datasets/MarsFet/derivatives/preprocessing"  

    MARSFET_ROSI_RESULTS = "/scratch/cmercier/results/rosi/"
    MARSFET_MESO_ROSI = "/scratch/cmercier/results/rosi"
    
    MARSFET_DATABASE = "/scratch/cmercier/code/pyrecon/bd_chapter4.csv"

    output_data = MARSFET_MESO_ROSI
    stacks_path = MARSFET_DATAPATH
    job_path = MARSFET_ROSI_RESULTS
    csv_file =  MARSFET_DATABASE
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
            #if subject== "sub-0051" and session == "ses-0061":
            #if subject== "sub-0675" and session == "ses-0801":
            #subject in sub_list and session in ses_list :
            #subject== "sub-0001" and session == "ses-0001":
            if subject== "sub-0675" and session == "ses-0801":
            #subject in sub_list and session in ses_list :
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
                job = os.path.join(job_path,subject,session,'rosi_alone')
                list_masks = ' '.join(str(list_masks) for list_masks in list_masks)
                dir_out = os.path.join(output_data, subject, session,'res_alone/res_mse.joblib.gz')
                print(dir_out)
                classifier = "my_model_nmse_dice_inter.pickle"
                cmd_os = "--filenames " + job 
                cmd_os += " --filenames_masks " + list_masks 
                cmd_os += " --output " + dir_out
                cmd_os += " --classifier " + classifier
                #if True :
                #not os.path.exists(os.path.join(dir_out,'1.nii.gz')):
                if subject in sub_list and session in ses_list :
                #subject == "sub-0051" and session == "ses-0061" :
                    print('input_slices:',list_stacks)
                    print('dir_output:',dir_out)
                    
                    cmd = (
                        "sbatch"
                        + " "
                        + "/scratch/cmercier/code/pyrecon/ROSI/utils/slurm/ms.slurm"
                        + " "
                        + '"'
                        + cmd_os
                        + '"'
                        + " "
                        )
                    
                    os.system(cmd)

