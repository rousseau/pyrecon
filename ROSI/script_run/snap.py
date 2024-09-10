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
    reorient = True
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
            if subject in sub_list and session in ses_list :
            #== "sub-0009" and session == "ses-0012":
            #
            #
            #== "sub-0002" and session == "ses-0002":

                path_to_mask = os.path.join(output,'nesvor',subject,session,"volume_mask.nii")
                path_to_volume = os.path.join(output,'nesvor',subject,session,"volume_rosi_outliers.nii")
               
                if  os.path.exists(path_to_volume):

                    #save in nisnap simple visualisation
                    #code issue de MarsFet/fet-processing
                    prefix_output = os.path.join(output,'snap2','all',subject,session)
                    figsize = {'x': (18, 4), 'y': (18, 4), 'z': (18, 5)}
                    if not os.path.exists(prefix_output):
                        os.makedirs(prefix_output)
                    snap=os.path.join(prefix_output, subject + "_" + session + "_" +  ".png")
                    #os.path.join(prefix_output,"snap.png")
                    image_shape = nib.load(path_to_volume).shape
                    data = np.ones(image_shape)
                    output_mask = nib.Nifti1Image(data,nib.load(path_to_volume).affine)
                    nib.save(output_mask,path_to_mask)
                    if not os.path.exists(snap):
                        done = 0
                        d_max = 150
                        step = 20
                        if reorient:
                            data = np.swapaxes(data,0,2)
                        while (done < 1) and (d_max > 20):
                            try:
                                slices = {'x': list(range(30, d_max, step)),'y': list(range(60, d_max, step)),'z': list(range(40, d_max, step))}
                                nisnap.plot_segment(path_to_mask,slices=slices,bg=path_to_volume,opacity=0,savefig=snap,contours=False,samebox=True,figsize=figsize)
                                done=1
                            except Exception as e:
                                print(e)
                                d_max = d_max - 20
                                step = step-5
                                print("d_max is now set to ", d_max)
                    



                    

