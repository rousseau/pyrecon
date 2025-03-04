#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 09:20:15 2022

@author: mercier
"""

from nibabel import Nifti1Image,load
import numpy as np
from rosi.reconstruction.rec_ebner import sorted_alphanumeric
from .sliceObject import SliceObject
from .tools import distance_from_mask
from .sliceObject import SliceObject
import os

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
        
        if ~(np.all(slice_mask==0)): #Check that the slice mask is not null. If it is, the slice will be deleted because it has no interest.
            
           
            mz = np.eye(4)

            #A translation in z is applied to the stack transformation to associate a transformation matrix with each slice : 
            #R_f(k) R_k,2t3d
            mz[2,3]= zi
            slice_transformation = stack.affine @ mz 
            
            new_slice = Nifti1Image(slice_value.copy(),slice_transformation)
            new_object = SliceObject(new_slice,slice_mask.copy(),index_stack,zi,index_volume)
            OutputList.append(new_object)
        else :
            print('mask_nul')
   
    return OutputList


def loadStack(fileImage : str,
              fileMask : str) -> (Nifti1Image,Nifti1Image):
    
    """
    Load stack and mask from files using nibabel library
    """
    
    stack = load(fileImage)
    #stack = stack_original.copy()
    if fileMask == None: ##If the mask wasn't provided, one is created covering the entire image.
          fileMask = np.ones(stack.get_fdata().shape)
          stmask = Nifti1Image(fileMask,stack.affine,dtype=np.int16)
    else :
          stmask = load(fileMask).get_fdata()
          print(stmask[np.where(stmask>0)])
          #check that the mask is a binary image
          #
          stmask = np.round(stmask)
          print(stmask[np.where(stmask>0)])
          stmask = np.array(stmask,dtype=np.int64)
          print(stmask[np.where(stmask>0)])
          data = stmask.reshape(-1)
          stmask = Nifti1Image(stmask,stack.affine,dtype=np.int16)
          print(data[data<1])
          #)

          if not (np.all((data==0)|(data==1))):
               raise Exception('The mask is not a binary image')
    return stack,stmask


def loadFromdir(dir_input):
     
     """
     Create a listofSlice from the direcory containing all the slices
     """

     list_file = sorted_alphanumeric([file for file in os.listdir(dir_input) if os.path.isdir(dir_input)])
     OutputList = []
     list_stack_ortho = []
     list_image_size = []
     list_stack_zi = []
     for file in list_file:
          index_stack=0
          if not "mask" in file:
            slice_path = os.path.join(dir_input,file)
            nib_slice = load(slice_path)
            mask_path=os.path.join(dir_input,'mask_' + file)
            nib_mask = load(mask_path)
            
            new_stack = False
            i=0
            for i in range(0,len(list_stack_ortho)):
                nx = nib_slice.affine[0:3,0]
                ny = nib_slice.affine[0:3,1]
                nz = np.cross(nx,ny)
                vec_orthogonal = nz/np.linalg.norm(nz)
                vec_image_size = nib_slice.shape
                print(vec_orthogonal,list_stack_ortho[i],np.abs(np.dot(vec_orthogonal,list_stack_ortho[i])))
                if np.abs(np.dot(vec_orthogonal,list_stack_ortho[i])) > 0.7 and vec_image_size == list_image_size[i]:
                    index_stack = i
                    break
                else : 
                    if i == (len(list_stack_ortho)-1):
                        print('ici')
                        new_stack = True
                        
            if new_stack or len(list_stack_ortho)==0 : 
                nx = nib_slice.affine[0:3,0]
                ny = nib_slice.affine[0:3,1]
                nz = np.cross(nx,ny)
                vec_orthogonal = nz/np.linalg.norm(nz)

                list_stack_ortho.append(vec_orthogonal)
                list_image_size.append(nib_slice.shape)
                list_stack_zi.append(1)
                index_stack = len(list_stack_zi)-1

            zi = list_stack_zi[index_stack]+1
            list_stack_zi[index_stack]=zi
            print(list_stack_zi)
            index_volume = index_stack
            print(file,index_stack,zi)
            new_object = SliceObject(nib_slice,nib_mask.get_fdata(),index_stack,zi,index_volume)
            OutputList.append(new_object)

     return OutputList



def convert2ListSlice(dir_nomvt,dir_slice,slice_thickness,set_of_affines):
    """
    Function to convert corrected slices output by svort or nesvor into sliceObjects.

    Inputs:
    dir_slice: Path to the directory containing motion-corrected slices (output of Svort/Nesvor).
    dir_nomvt: Path to the directory containing slices without simulated motion.
    set_of_affines: List of affine matrices representing the original low-resolution (LR) image transformations.
    slice_thickness: Thickness of the slices


    Outputs:
    listnomvt: List of SliceObject instances generated from the dir_nomvt directory.
    listSlice:  List of SliceObject instances generated from the dir_slice directory.

    N.B : The dir_nomvt directory is required to determine the slice index and stack index for each SliceObject during conversion.
    """
    #print(dir_nomvt)
    ##
    list_file = sorted_alphanumeric(os.listdir(dir_slice))
    #print(list_file)

    listSlice=[];listnomvt=[]
    index=np.zeros(3,np.int32)
    for file in list_file:
        #check that the file is a slice
        
        if not 'mask' in file and not 'image' in file and not 'volume' in file : ##for svrtk you would need to have "slice"
            ##else do nothing
            print('file_name :',file)
            slice_data=nib.load(dir_slice + '/' + file)
            mask_data=nib.load(dir_slice + '/mask_' + file)
            slice_nomvt = nib.load(dir_nomvt + '/' + file)
            mask_slice=nib.load(dir_slice + '/mask_' + file)
            print("affine equal ? ",np.allclose(slice_data.affine,slice_nomvt.affine,1e-1))
                #print(slice_data.affine - slice_nomvt.affine)
            print(slice_data.affine)
            print(slice_nomvt.affine)
                
            T0 = slice_data.affine


            num_stack=which_stack(slice_nomvt.affine,slice_thickness) 
                #print(slice_data.affine)
            stack_affine=set_of_affines[num_stack]
            
            i_slice=where_in_the_stack(stack_affine,slice_nomvt.affine,num_stack)
            slicei = SliceObject(slice_data,mask_data.get_fdata(),num_stack,i_slice, num_stack)
            slicen = SliceObject(slice_nomvt,mask_slice.get_fdata(),num_stack,i_slice, num_stack)
                
            listSlice.append(slicei)
            listnomvt.append(slicen)
    return listnomvt,listSlice