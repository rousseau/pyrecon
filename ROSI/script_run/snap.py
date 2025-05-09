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

    MARSFET_MESO_RESULTS ="/home/cmercier/results/"

    MARSFET_MESO_ms_INIT = "/home/cmercier/results/ms"
    
    MARSFET_DATABASE = "/scratch/cmercier/code/pyrecon/bd_chapter4.csv"

    output = MARSFET_MESO_RESULTS
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

                path_to_mask = os.path.join(output,'nesvor',subject,session,"volume_mask.nii")
                #path_to_reconstruction = "/scratch/gauzias/data/datasets/MarsFet/derivatives/srr_reconstruction/niftymic-0.8-iso/"
                path_to_volume = os.path.join(output,'nesvor',subject,session,"volume_rosi_tru.nii")
                #path_to_volume = os.path.join(path_to_reconstruction,subject,session,"haste/default_reconst/","%s_%s_acq-haste_rec-nesvor_T2w.nii.gz" %(subject,session))
                #path_to_volume = os.path.join(path_to_reconstruction,subject,session,"haste/recon_template_space","srr_template.nii.gz")
                #path_to_mask = os.path.join(path_to_reconstruction,subject,session,"haste/recon_template_space","srr_template_mask.nii.gz")

                if  os.path.exists(path_to_volume):

                    print(path_to_volume)
                    #save in nisnap simple visualisation
                    #code issue de MarsFet/fet-processing
                    prefix_output = os.path.join(output,'snap2','svrtk',subject,session)
                    figsize = {'x': (18, 4), 'y': (18, 4), 'z': (18, 5)}
                    if not os.path.exists(prefix_output):
                        os.makedirs(prefix_output)
                    snap=os.path.join(prefix_output, subject + "_" + session + "_" +  ".png")
                    print(snap)
         
                    data=nib.load(path_to_mask).get_fdata()
                
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
                else : 
                    print(path_to_volume)



                    

