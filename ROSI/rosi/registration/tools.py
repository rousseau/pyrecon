#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 10:00:47 2021

@author: mercier
"""
import array
import numpy as np
from numpy import linspace
from numpy import concatenate
from numpy import zeros
from numpy import sum
from scipy.ndimage.filters import gaussian_filter
from nibabel import Nifti1Image
from scipy.ndimage import distance_transform_cdt
from .sliceObject import SliceObject
from numba import jit


def apply_gaussian_filtering(listOfSlice: 'list[SliceObject]' ,
                             sigma : float) -> list:
    """
    Apply a gaussian filter to each slice of the list
    """
    
    #Initialization
    blurlist=[]
    
    for i_slice in range(len(listOfSlice)):
        
        slicei=listOfSlice[i_slice].copy()
        slice_value=slicei.get_slice().get_fdata()
        blur_value = gaussian_filter(slice_value,sigma=sigma) #applied gaussian filter to values of the slice
        blur_slice=Nifti1Image(blur_value, slicei.get_slice().affine) #create a new slice from the blur values
        slicei.set_slice(blur_slice)
        blurlist.append(slicei)
    
    return blurlist


def separate_slices_in_stacks(listOfSlice : 'list[SliceObject]') -> (np.array('list[SliceObject]'),np.array('list[SliceObject]')):
    """
    Separate a list of SliceObject into n list corresponding to n stacks
    """
    
    volume_index=[]; volumes=[]; volume_masks=[]

    for slicei in listOfSlice:
        stack_of_slicei = slicei.get_indexVolume()
        print(stack_of_slicei)

        if stack_of_slicei in volume_index: #if we already encounter a slice from this volume
            index_orientation = volume_index.index(stack_of_slicei)
            volumes[index_orientation].append(slicei) #add the slice to the corresponding list ! :)
            volume_masks[index_orientation].append(slicei.get_mask())
        
        else: #we never encountered a slice from this volume before :(
            volume_index.append(stack_of_slicei) 
            volumes.append([]) #create a new list
            volume_masks.append([])
            index_orientation = volume_index.index(stack_of_slicei) #add the slice to this new list ! 
            volumes[index_orientation].append(slicei)
            volume_masks[index_orientation].append(slicei.get_mask())
                
    return volumes, volume_masks

def distance_from_edge(volume : Nifti1Image) -> np.array:
    """
    Chamfer distance from the edge to the mask
    """
    values=volume.get_fdata()
    data=values.reshape(-1)
    if not (np.all((data==0)|(data==1))): #make sure the image is binary
        raise Exception("The mask is not a binary image !!")  
    
    inv_chamfer_distance = distance_transform_cdt(values)
    
    return inv_chamfer_distance

def distance_from_mask(volume : array) -> array:
    """
    chamfer distante from the to the edge
    """
    data=volume.reshape(-1)
    if not (np.all((data==0)|(data==1))): #make sure the image is binary
        raise Exception("The mask is not a binary image !!")  
    
    inv_chamfer_distance = distance_transform_cdt(1-volume)

    return inv_chamfer_distance


def computeMaxVolume(listSlice : 'list[SliceObject]') -> float :
    """
    from a list of sliceObject, compute the maximum volume of the mask. This is use to normalize the intersection in the cost function
    """
    
    Vmx =  np.sum([np.sum(slicei.get_mask()) for slicei in listSlice])

    return Vmx

@jit(nopython=True,fastmath=True)
def somme(index):
    return sum(index)

@jit(nopython=True,fastmath=True)
def line(start,stop,num):
    return linspace(start,stop,num)


def normalization(listOfSlices):
    """
    Normalized the images to a standard normal distribution
    """
   
   
    data  = concatenate([s.get_slice().get_fdata().reshape(-1) for s in listOfSlices])
    mask = concatenate([s.get_mask().reshape(-1) for s in listOfSlices])
    var = data[mask>0]
    mean = np.mean(var)
    std = np.std(var)
  
    normalised_list = []
    
    for k in listOfSlices:
        slice_k = k.get_slice().get_fdata() 
        X,Y,Z = slice_k.shape
        
        normalised_slice = zeros((X,Y,1))
        for x in range(X):
            for y in range(Y):
                normalised_slice[x,y,0] = (slice_k[x,y,0] - mean)/std
        newNifti = Nifti1Image(normalised_slice, k.get_slice().affine)
        k.set_slice(newNifti)
        normalised_list.append(k)

   
    return normalised_list


def same_order(listSlice,listnomvt,transfo):
    """
    The function consider a list Of SliceObject, a the same slice but without motion 
    It's retrun the different lists in the same order

    Inputs:
    listSlice: A list of sliceObject instance, representing the 2D slices with simulated motion.
    listnomvt: A list of sliceObject instance, representing the 2D slices without simulated motion.
    transfo: A list of the theorical simulated transformation

    """
    
    img,_=separate_slices_in_stacks(listSlice)
    img=np.array(img,dtype=list)
    print(img.shape)
    nomvt,_=separate_slices_in_stacks(listnomvt)
    nomvt=np.array(nomvt,dtype=list)
    print(nomvt.shape)
    vectzimg = np.zeros(len(img))
    vectznomvt = np.zeros(len(img))
    #affine matrix are supposed to be the same, but in different order
    for image in range(0,len(img)):
            
            timg=img[image][0].get_slice().affine
            nximg=timg[0:3,0].copy()
            nyimg=timg[0:3,1].copy()
            nx=nximg/np.linalg.norm(nximg)
            ny=nyimg/np.linalg.norm(nyimg)
            nz=np.abs(np.cross(nx,ny))
            vectzimg[image]=np.argmax(nz)

            tnomvt=nomvt[image][0].get_slice().affine
            nxnomvt=tnomvt[0:3,0].copy()
            nynomvt=tnomvt[0:3,1].copy()
            nx=nxnomvt/np.linalg.norm(nxnomvt)
            ny=nynomvt/np.linalg.norm(nynomvt)
            nz=np.abs(np.cross(nx,ny))
            vectznomvt[image]=np.argmax(nz)

    minimg = np.argsort(vectzimg)
    minnomvt = np.argsort(vectznomvt)

    return img[minimg],nomvt[minnomvt],transfo[minnomvt]

