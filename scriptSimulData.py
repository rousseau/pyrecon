#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:25:56 2022

@author: mercier
"""

from simul3Ddata import extract_mask,simulateMvt
import nibabel as nib
from load import loadSlice
import numpy as np
import os


#script to simulate LR image with motion from an HR image

#The function 'simulateMvt' simulates a LR image with inter-slice motion from an HR image.
#SimulateMVt take as parameters : the original HRImage, range motion for rotation, range motion for translation, upsampling parameters (to choose interslice resolution of LR image), image orientation, binary image corresponding to the mask and a boolean (Set to false if you don't want motion)
#And return : the LR image, mask of the LR image, parameters of transformation for each slices, rigid transformation for each slices.

HRnifti = nib.load('/home/mercier/Documents/donnee/DHCP/image_2.nii.gz') #3D isotropic image
Mask = nib.load('/home/mercier/Documents/donnee/DHCP/mask_2.nii.gz') #mask associated to the image

binaryMask = extract_mask(Mask) #convert mask to a biniary mask


for i in range(1):
    
    i=i+1
    print(i)

    #os.mkdir('/home/mercier/Documents/donnee/test/Grand5/')
    
    
    LrAxNifti,AxMask,paramAx,transfoAx = simulateMvt(HRnifti,[-5,5],[-5,5],6,'axial',binaryMask.get_fdata(),True)#create an axial volume
    LrCorNifti,CorMask,paramCor,transfoCor = simulateMvt(HRnifti,[-5,5],[-5,5],6,'coronal',binaryMask.get_fdata(),True) #create a coronal volume
    LrSagNifti,SagMask,paramSag,transfoSag = simulateMvt(HRnifti,[-5,5],[-5,5],6,'sagittal',binaryMask.get_fdata(),True)#create a sagittal volume

    nib.save(LrAxNifti,'/home/mercier/Documents/donnee/test/Moyen2/LrAxNifti_moyen2.nii.gz') #save images, masks, parameters and global transformations
    nib.save(AxMask,'/home/mercier/Documents/donnee/test/Moyen2/AxMask_moyen2.nii.gz')
    np.save('/home/mercier/Documents/donnee/test/Moyen2/paramAx_moyen2.npy',paramAx)
    np.save('/home/mercier/Documents/donnee/test/Moyen2/transfoAx_moyen2.npy',transfoAx)
    
    nib.save(LrCorNifti,'/home/mercier/Documents/donnee/test/Moyen2/LrCorNifti_moyen2.nii.gz')
    nib.save(CorMask,'/home/mercier/Documents/donnee/test/Moyen2/CorMask_moyen2.nii.gz')
    np.save('/home/mercier/Documents/donnee/test/Moyen2/paramCor_moyen2.npy',paramCor)
    np.save('//home/mercier/Documents/donnee/test/Moyen2/transfoCor_moyen2.npy',transfoCor)
    
    nib.save(LrSagNifti,'/home/mercier/Documents/donnee/test/Moyen2/LrSagNifti_moyen2.nii.gz')
    nib.save(SagMask,'/home/mercier/Documents/donnee/test/Moyen2/SagMask_moyen2.nii.gz')
    np.save('/home/mercier/Documents/donnee/test/Moyen2/paramSag_moyen2.npy',paramSag)
    np.save('/home/mercier/Documents/donnee/test/Moyen2/transfoSag_moyen2.npy',transfoSag)




