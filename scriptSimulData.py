#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:25:56 2022

@author: mercier
"""

from simul3Ddata import extract_mask,simulateMvt
import nibabel as nib
from registration import loadSlice
import numpy as np

#script to simulate movment in an image


HRnifti = nib.load('/home/mercier/Documents/donnee/dhcp/sub-CC00060XX03_ses-12501_desc-restore_T2w.nii.gz') #3D isotropic image
Mask = nib.load('/home/mercier/Documents/donnee/dhcp/sub-CC00060XX03_ses-12501_desc-fusion_space-T2w_dseg.nii.gz') #mask associated to the image

binaryMask = extract_mask(Mask) #convert mask to a biniary mask

#level of movment :
#small : [-3,3],[-3,3]
#medium : [-5,5],[-5,5]
#large : [-10,10],[-10,10]

LrAxNifti,AxMask,paramAx,transfoAx = simulateMvt(HRnifti,[-3,3],[-3,3],'axial',binaryMask.get_fdata()) #create an axial volume
LrCorNifti,CorMask,paramCor,transfoCor = simulateMvt(HRnifti,[-3,3],[-3,3],'coronal',binaryMask.get_fdata()) #create a coronal volume
LrSagNifti,SagMask,paramSag,transfoSag = simulateMvt(HRnifti,[-3,3],[-3,3],'sagittal',binaryMask.get_fdata()) #create a sagittal volume

nib.save(LrAxNifti,'/home/mercier/Documents/donnee/dhcp/Petit/LrAxNifti_petit.nii.gz') #save images, masks, parameters and global transformations
nib.save(AxMask,'/home/mercier/Documents/donnee/dhcp/Petit/AxMask_petit.nii.gz')
np.save('/home/mercier/Documents/donnee/dhcp/Petit/paramAx_petit.npy',paramAx)
np.save('/home/mercier/Documents/donnee/dhcp/Petit/transfoAx_petit.npy',transfoAx)

nib.save(LrCorNifti,'/home/mercier/Documents/donnee/dhcp/Petit/LrCorNifti_petit.nii.gz')
nib.save(CorMask,'/home/mercier/Documents/donnee/dhcp/Petit/CorMask_petit.nii.gz')
np.save('/home/mercier/Documents/donnee/dhcp/Petit/paramCor_petit.npy',paramCor)
np.save('/home/mercier/Documents/donnee/dhcp/Petit/transfoCor_petit.npy',transfoCor)

nib.save(LrSagNifti,'/home/mercier/Documents/donnee/dhcp/Petit/LrSagNifti_petit.nii.gz')
nib.save(SagMask,'/home/mercier/Documents/donnee/dhcp/Petit/SagMask_petit.nii.gz')
np.save('/home/mercier/Documents/donnee/dhcp/Petit/paramSag_petit.npy',paramSag)
np.save('/home/mercier/Documents/donnee/dhcp/Petit/transfoSag_petit.npy',transfoSag)




