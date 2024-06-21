#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:20:15 2022

@author: mercier
"""

from nibabel import Nifti1Image,load
import numpy as np
from .sliceObject import SliceObject
from .tools import distance_from_mask
from .sliceObject import SliceObject

def convert2Slices(stack : Nifti1Image,
              mask : Nifti1Image,
              listOfSlices : 'list[SliceObject]',
              index_stack : int,
              index_volume : int) -> list:
    
    """
    Take an LR volume as input and convert each slice into a sliceObject. Then, add each slice to the input list
    (listOfSlices).
    """
    
    OutputList = listOfSlices.copy()

    if mask == None: #If the mask wasn't provided, one is created covering the entire volume.
        mask = Nifti1Image(np.ones(stack.get_fdata().shape), stack.affine)
    
    X,Y,Z = stack.shape
    slice_value = np.zeros((X,Y,1))
    slice_mask = np.zeros((X,Y,1))
    
    res=min(mask.header.get_zooms())
    
    for zi in range(Z): #for each slices in the stack
        
        slice_value[:,:,0] = stack.get_fdata()[:,:,zi]
        slice_mask[:,:,0] = mask.get_fdata()[:,:,zi].astype(int)
        slice_mask[np.isnan(slice_mask)]=0
        
        #The slice is linearly cropped according to the distance to the mask
        dist = distance_from_mask(slice_mask)*res
        decrease=np.linspace(1,0,6,dtype=float)
        index=0
        
        for index_dist in range(4,10):
             slice_value[np.where(dist>index_dist)] = decrease[index]*slice_value[np.where(dist>index_dist)]
             index+=1
 
        if ~(np.all(slice_mask==0)): #Check that the slice mask is not null. If it is, the slice will be deleted because it has no interest.
           
            mz = np.eye(4)

            #A translation in z is applied to the stack transformation to associate a transformation matrix with each slice : 
            #R_f(k) R_k,2t3d
            mz[2,3]= zi
            slice_transformation = stack.affine @ mz 
            
            new_slice = Nifti1Image(slice_value.copy(),slice_transformation)
            new_object = SliceObject(new_slice,slice_mask.copy(),index_stack,zi,index_volume)
            OutputList.append(new_object)
   
    return OutputList


def loadStack(fileImage : str,
              fileMask : str) -> (Nifti1Image,Nifti1Image):
    
    """
    Load stack and mask from files using nibabel library
    """
    
    stack = load(fileImage)
    if fileMask == None: ##If the mask wasn't provided, one is created covering the entire image.
          fileMask = np.ones(stack.get_fdata().shape)
          stmask = Nifti1Image(fileMask,stack.affine)
    else :
          stmask = load(fileMask)
          #check that the mask is a binary image
          data = stmask.get_fdata().reshape(-1)
          data = np.round(data)
          data = np.array(data.tolist(),dtype=np.int64)

          if not (np.all((data==0)|(data==1))):
               raise Exception('The mask is not a binary image')
    return stack,stmask
