import os
import re
import pathlib
import subprocess
import csv


if __name__ == "__main__":


    MARSFET_DATAPATH = "/scratch/gauzias/data/datasets/MarsFet/derivatives/preprocessing"  

    MARSFET_MESO_ROSI = "/scratch/cmercier/results/rosi/"

    MARSFET_MESO_RESULTS = "/scratch/cmercier/results/"
    
    MARSFET_DATABASE = "/scratch/cmercier/code/pyrecon/marsfet_latest_participants.csv"

    job_res = MARSFET_MESO_ROSI
    output = MARSFET_MESO_RESULTS
    stacks_path = MARSFET_DATAPATH
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
            if subject == "sub-0002" and session == "ses-0002":
            #in sub_list and session in ses_list :
            #== "sub-0002" and session == "ses-0002":
                joblib_path = os.path.join(job_res,subject, session, 'res.joblib.gz')
                input_stacks = os.path.join(stacks_path,subject,session)
                list_stacks=[]
                list_masks=[]
                for file in input_stacks:
                    if file.endswith("denoised_T2w.nii.gz") and 'haste' in file:
						#stack = os.path.join(dir_reconst, subject+ "_"+ session + "_"+ "acq-"+ sequence+ "_"+ "run" + "-" + serie + "_desc-denoised_T2w.nii.gz")
                        path_to_file = os.path.join('/data',subject,session,file)
                        list_stacks.append(path_to_file)
                        run = file.find("run") #find the number of the run to make sure the stack is associated with its corresponding mask
                        num_index = run + 4
                        num = "run-%s" %(file[num_index])
                        print("number run",num)
                        for file in input_stacks:
                            if file.endswith("brainmask_T2w.nii.gz") and 'haste' in file and num in file:
                                path_to_file = os.path.join('/data',subject,session,file)
                                list_masks.append(path_to_file)
                                break
                list_stacks = ' '.join(str(list_stacks) for list_stacks in list_stacks)
                list_masks = ' '.join(str(list_masks) for list_masks in list_masks)
                output_svort = os.path.join(output,'rosi', 'slices', subject, session)
                output_nesvor = os.path.join(output,'nesvor',subject,session)
                output_nesvor_slices = os.path.joint(output,'nesvor','slices',subject,session)

                if os.path.exists(joblib_path):
                    cmd_os_1 = " --input-stacks " + list_stacks
                    cmd_os_1 += " --input-mask " + list_masks
                    cmd_os_1 += " --output " + output_svort
                    cmd_os_1 += " --results " + joblib_path

                    cmd_os_2 =  " --input_slices " + output_svort
                    cmd_os_2 += " --output-volume " + output_nesvor
                    cmd_os_2 += " --output-slices " + output_nesvor_slices
                    cmd_os_2 += " --registration none "
                    cmd_os_2 += " --no-transformation-optimization "

                    cmd = (
                        "sbatch"
                        + " "
                        + "/scratch/cmercier/code/pyrecon/ROSI/utils/slurm/nesvor.slurm"
                        + " "
                        + '"'
                        + cmd_os_1
                        + '"'
                        + " "
                        + '"'
                        + cmd_os_2
                        + "'"
                        + " "
                        + MARSFET_DATABASE
                        )
                    
                    #os.system(cmd)
                    print(cmd)

