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
    
    MARSFET_DATABASE = "/scratch/cmercier/code/pyrecon/marsfet_latest_participants.csv"

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
            if subject == "sub-0002" and session == "ses-0002":
            #in sub_list and session in ses_list :
            #== "sub-0002" and session == "ses-0002":
                joblib_path = os.path.join(job_res,subject, session, 'res.joblib.gz')
                input_slices = os.path.join(slices_path,subject,session,'res')
                #print(input_sl)
                #print("input_stacks",input_stacks)
            
                
                output_svort = os.path.join(output,'nesvor', 'rosi_slices', subject, session)
                output_svort_mask = os.path.join(output,'nesvor', 'rosi_slices_mask', subject, session)
                output_svort_similarity = os.path.join('/data','nesvor', 'rosi_slices', subject, session)
                output_nesvor = os.path.join('/data','nesvor',subject,session,"volume.nii")
                path_to_mask = os.path.join(output,'nesvor',subject,session,"volume_mask.nii")
                path_to_volume = os.path.join(output,'nesvor',subject,session,"volume.nii")
                output_nesvor_slices = os.path.join(output,'nesvor','slices',subject,session)

                if os.path.exists(joblib_path):
                    cmd_os_1 = " --input_slices " + input_slices
                    cmd_os_1 += " --output " + output_svort
                    cmd_os_1 += " --output_mask " + output_svort_mask
                    cmd_os_1 += " --results " + joblib_path

                    cmd_os_2 =  " --input-slices " + output_svort
                    cmd_os_2 += " --output-volume " + output_nesvor
                    cmd_os_2 += " --output-slices " + output_nesvor_slices
                    cmd_os_2 += " --registration none "
                    cmd_os_2 += " --no-transformation-optimization "

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



                    #save in nisnap simple visualisation
                    prefix_output = os.path.join(output,'nesvor',subject,session)
                    snap_ax=os.path.join(prefix_output,"snap_axial.png")
                    snap_cor=os.path.join(prefix_output,"snap_cor.png")
                    snap_sag=os.path.join(prefix_output,"snap_sagital.png")
                    image_shape = nib.load(output_nesvor).shape()
                    data = np.ones(image_shape)
                    output_mask = nib.Nifti1Image(data,nib.load(output_nesvor).affine)
                    nib.save(output_mask,path_to_mask)
                    nisnap.plot_segment(path_to_mask,axes='x',bg=path_to_volume,opacity=20,savefig=snap_ax,contours=False)
                    nisnap.plot_segment(path_to_mask,axes='y',bg=path_to_volume,opacity=20,savefig=snap_cor,contours=False)
                    nisnap.plot_segment(path_to_mask,axes='z',bg=path_to_volume,opacity=20,savefig=snap_sag,contours=False)
                    #print(cmd)

