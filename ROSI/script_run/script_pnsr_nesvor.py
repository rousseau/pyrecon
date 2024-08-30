

import os
import ants
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import nibabel as nib
import matplotlib.figure as fig
import matplotlib.pyplot as plt

path_to_reference = "/home/mercier/Documents/donnee/DHCP/"
path_to_moving = "/home/mercier/Documents/res/nesvor_alone/"

def EQM(reference_image,moving_image):
    X,Y,Z = reference_image.shape
    EQM = (reference_image-moving_image)**2
    EQM = EQM[np.where(EQM>0)]
    sum_EQM = np.sum(EQM)
    EQM = sum_EQM/len(EQM)
    return EQM
    
def PSNR(reference_image,moving_image):
    moving_image=moving_image
    eqm = EQM(reference_image,moving_image)
    d=1
    #np.max([np.max(reference_image), np.max(moving_image)])-np.min([np.min(reference_image), np.min(moving_image)])
    print(np.max(reference_image))
    print(np.max(moving_image))
    #d=np.max(reference_image)-np.min(reference_image)
    #res = 10*np.log10(d**2/eqm)
    res = psnr(reference_image,moving_image,data_range=d)
    return res

def SSIM(reference_image,moving_image):
    # 5. Compute the Structural Similarity Index (SSIM) between the two images
    (score, diff) = ssim(reference_image, moving_image, full=True)
    diff = (diff * 255).astype("uint8")
    return score
    

motion_range = ["Petit"]
#["Grand","Moyen","Petit","tres_petit"]
nb_data = range(1,5)

plist = []
slist = []
for motion in motion_range : 
    for i in nb_data : 
        reference_path = os.path.join(path_to_reference,'image_%d.nii.gz' %(i))
        mask_path = os.path.join(path_to_reference,'binmask_%d.nii.gz' %(i))
        #moving_path = os.path.join(path_to_moving,'%s%d/lamb0/recon_subject_space/srr_subject.nii.gz' %(motion,i))
        moving_path = os.path.join(path_to_moving,'%s%d/volume.nii.gz' %(motion,i))
        print(moving_path)
        if os.path.exists(moving_path):
            #neg_det = nib.load(reference_path)
            #pos_det = nib.as_closest_canonical(neg_det)
            #new_reference_path = os.path.join(path_to_reference,'image_%d_fliped.nii.gz' %(i))
            #nib.save(pos_det,new_reference_path)
            reference_image = ants.image_read(reference_path)
            reference_mask = ants.image_read(mask_path)
            moving_image = ants.image_read(moving_path)
            transform = ants.registration(reference_image*reference_mask,moving_image,'Rigid')
            moving_image_registered = ants.apply_transforms(reference_image*reference_mask,moving_image,transformlist=transform['fwdtransforms'])
            numpy_moving_image = moving_image_registered.numpy()
            numpy_reference_image = reference_image.numpy()
            numpy_reference_mask = reference_mask.numpy()
            index = np.where(numpy_reference_mask>0)
            numpy_moving_image = numpy_moving_image * numpy_reference_mask
            numpy_reference_image = numpy_reference_image * numpy_reference_mask
            #print(index)
            numpy_reference_image = (numpy_reference_image - np.min(numpy_reference_image)) / (np.max(numpy_reference_image) - np.min(numpy_reference_image))
            numpy_moving_image = (numpy_moving_image - np.min(numpy_moving_image)) / (np.max(numpy_moving_image) - np.min(numpy_moving_image))
            d_image = np.abs(numpy_reference_image-numpy_moving_image)
            save_path = os.path.join(path_to_reference,'difference_image.nii.gz')
            ants_d = moving_image_registered.new_image_like(d_image)
            #ants.image_write(ants_d,save_path)
            p = PSNR(numpy_reference_image,numpy_moving_image)
            ssim_res = SSIM(numpy_reference_image,numpy_moving_image)
            plist.append(p)
            slist.append(ssim_res)