#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:20:15 2022

@author: mercier
"""

from nibabel import Nifti1Image,load
import numpy as np
from sliceObject import SliceObject

def loadSlice(img,mask,listSlice,orientation,indexImage):
    """
    Create sliceObject from the slice and the mask. When the mask associated to the slice is completly dark, the slice is removed.
    """
    
    if mask == None:
        mask = Nifti1Image(np.ones(img.get_fdata().shape), img.affine)
    
    X,Y,Z = img.shape
    slice_img = np.zeros((X,Y,1))
    slice_mask = np.zeros((X,Y,1))
    
    print('z',img.shape)
    
    for zi in range(Z): #Lecture des images
        slice_img[:,:,0] = img.get_fdata()[:,:,zi]
        slice_mask[:,:,0] = mask.get_fdata()[:,:,zi]
        slice_mask[np.isnan(slice_mask)]=0
        if ~(np.all(slice_mask==0)):
            #print(zi)
            mz = np.eye(4)
            mz[2,3]= zi
            sliceaffine = img.affine @ mz
            nifti = Nifti1Image(slice_img.copy(),sliceaffine)
            c_im1 = SliceObject(nifti,slice_mask.copy(),orientation,zi,indexImage)
            listSlice.append(c_im1)
        #else :
        #    print("slice %d removed" %(zi))
        
    return listSlice

def loadimages(fileImage,fileMask):
    """
    load images and mask from files

    """
    
    im = load(fileImage)
    if fileMask == None:
          fileMask = np.ones(im.get_fdata().shape)
          inmask = Nifti1Image(fileMask,im.affine)
    else :
          inmask = load(fileMask)
    return im,inmask
