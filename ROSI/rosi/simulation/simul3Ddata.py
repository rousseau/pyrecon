#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:49:35 2022

@author: mercier
"""

import numpy as np
#import nibabel as nib
from nibabel import Nifti1Image
from tools import rigidMatrix
from scipy.ndimage import map_coordinates
import random as rd
from ..registration import common_segment_in_image
from scipy.ndimage import distance_transform_cdt
from sliceObject import SliceObject


def simulate_motion_on_slices(listOfSlice : list(SliceObject),
                              rotation_min_max : np.array(2),
                              translation_min_max : np.array(2)) -> np.array:
    """
    The function create mouvement bteween the slices of a 3D mri image

    Inputs :
    listSlice : listSlice containing the original slices, with no movement

    Returns
    motion_parameters : the random motion parameters for each slices
    """

    nbSlice = len(listOfSlice)
    rangeAngle = rotation_min_max[1]-rotation_min_max[0]
    rangeTranslation = translation_min_max[1]-translation_min_max[0]
    motion_parameters = np.zeros((6,nbSlice))
    
    i = 0
    for s in listOfSlice: #add motion to each slice 
        x = 0
        a1 = rd.random()*(rangeAngle) - (rangeAngle)/2
        a2 = rd.random()*(rangeAngle) - (rangeAngle)/2
        a3 = rd.random()*(rangeAngle) - (rangeAngle)/2
        t1 = rd.random()*(rangeTranslation) - (rangeTranslation)/2
        t2 = rd.random()*(rangeTranslation) - (rangeTranslation)/2
        t3 = rd.random()*(rangeTranslation) - (rangeTranslation)/2
        x = np.array([a1,a2,a3,t1,t2,t3])
        s.set_parameters(x) #set the parameters of the slice
        motion_parameters[:,i]=x
        i=i+1
    
    return motion_parameters




#PSF is defined for the trough plan direction
def psf(x_0,x):  
   
    # #The PSF is a gaussian function
    FHWM = 1.0 #FHWM is equal to the slice thikness (for ssFSE sequence), 1 in voxel  (cf article Jiang et al.)
    sigma = FHWM/(2*np.sqrt((2*np.log(2))))
    res = (1/(sigma*np.sqrt(2*np.pi)))* np.exp((-(x-x_0)**2)/(2*(sigma**2)))

    return res

def extract_mask(NiftiMask : Nifti1Image) -> Nifti1Image:
   """
    Create a binary mask of the brain from a brain segmentation image. Useful on Dhcp segmentation.
   """
   mask=NiftiMask.get_fdata()
   X,Y,Z=NiftiMask.shape
   mask[mask==4]=0
   mask=mask>1e-2
   newMask = Nifti1Image(mask.astype(np.float),NiftiMask.affine)
    
   return newMask

def simulateMvt(image : Nifti1Image,
                AngleMinMax : np.array(2),
                TransMinMax : np.array(2),
                resolution : float,
                stack_num : int,
                mask : Nifti1Image,
                motion : bool = True) -> (Nifti1Image,Nifti1Image,np.array,np.array):
    """
    The function create 3 orthogonals low resolution volume with from 3D mri image

    Inputs : 
    image : 3D mri volume

    Returns :
    Volume: Volume in the choosen orientation


    """
    x,y,z = image.shape
     

    
    #create an LR image in axial orientation
    if stack_num=='axial':
        
        new_x=x;new_y=y;new_z=z
        current_thikness = image.header.get_zooms()[2]
        slice_thikness = resolution/current_thikness
        #matrix to change image orientation and sub-sample the image
        sub_sample_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,slice_thikness,0],[0,0,0,1]]) 

        
    #create an LR image in coronal orientation
    elif stack_num=='coronal':
        
        new_x=x;new_y=z;new_z=y
        current_thikness = image.header.get_zooms()[1]
        slice_thikness = resolution/current_thikness
        #matrix to change image orientation and sub-sample the image
        sub_sample_matrix = np.array([[1,0,0,0],[0,0,slice_thikness,0],[0,1,0,0],[0,0,0,1]])

        
        
    #create an LR image in sagittal orientation
    elif stack_num=='sagittal':
        
        new_x=z;new_y=y;new_z=x 
        current_thikness = image.header.get_zooms()[0]
        slice_thikness = resolution/current_thikness
        #matrix to change image orientation and sub-sample the image
        sub_sample_matrix = np.array([[0,0,slice_thikness,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])

   
    else :
        print('unkown orientation, choose between axial, coronal and sagittal')
        return 0
    
    slices_number=new_z//slice_thikness
    
    #Initialisation
    low_resolution_image = np.zeros((new_x,new_y,slices_number))
    low_resolution_mask = np.zeros((new_x,new_y,slices_number))
    parameters=np.zeros((slices_number,6))
    theorical_transformations=np.zeros((slices_number,4,4)) #equivalent to M_k
    
    slice_vect = np.linspace(0,slices_number-1,slices_number,dtype=int)
    
    hr_to_world = image.affine
    lr_to_world =  hr_to_world @ sub_sample_matrix

    
    #point coordinate for the PSF (in voxel coordinate)
    indexes_psf = np.linspace(-0.5,0.5,5)
    psf_values = psf(0,indexes_psf) 
    normalised_psf = psf_values/sum(psf_values)
    
    #For each slice of the low resolution image :
        #1/ Simulate rigid motion if we want some. If not, motion matrix is the identity
        #2/ Interpolate corresponding values from the HR image 
        #3/ Interpolate corresponding values for the HR mask image
        
    for i in slice_vect: 
        
        #No motion, motion matrix is the identity 
        if motion==False: 
            rigid_matrix = np.eye(4)
            parameters[i,:]= np.array([0,0,0,0,0,0])
            
        #Motion, random parameters are choosen. RangeAngle and RangeTranslation defined the level of motion we want to simulate   
        else : 
            RangeAngle=AngleMinMax[1]-AngleMinMax[0]
            RangeTranslation=TransMinMax[1]-TransMinMax[0]
            a1 = rd.random()*(RangeAngle) - (RangeAngle)/2
            a2 = rd.random()*(RangeAngle) - (RangeAngle)/2
            a3 = rd.random()*(RangeAngle) - (RangeAngle)/2
            t1 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            t2 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            t3 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            rigid_matrix = rigidMatrix([a1,a2,a3,t1,t2,t3])
            parameters[i,:]= np.array([a1,a2,a3,t1,t2,t3])
        
        #coordinate in the low resolution image, in homogenous coordinate. 
        coordinate_in_lr = np.zeros((4,new_x*new_y*5)) 
        
        #output of the interpolation
        interpolate = np.zeros(new_x*new_y*5) 
        slice_mask=np.zeros((new_x*new_y))
        
        #coordinate, in slice i in the LR image
        ii = np.arange(0,new_x) 
        jj = np.arange(0,new_y)
        iv,jv, zv = np.meshgrid(ii,jj,indexes_psf,indexing='ij')
         
        iv = np.reshape(iv, (-1))
        jv = np.reshape(jv, (-1))
        zv = np.reshape(zv, (-1))
        
        coordinate_in_lr[0,:] = iv
        coordinate_in_lr[1,:] = jv
        coordinate_in_lr[2,:] = i+zv
        coordinate_in_lr[3,:] = 1
        
        #center coordinate, of the slice, in image coordinate
        center_image = np.ones(4); center_image[0] = new_x//2; center_image[1] = new_y//2; center_image[2] = i; center_image[3]= 1
        
        #center coordinate, in voxel, in world coordinate
        center_world = lr_to_world @ center_image
        
        #translation matrix, from image corner to image center, in world coordinate
        corner_to_center = np.eye(4); corner_to_center[0:3,3]=-center_world[0:3]
        
        #translation matrix, from image center to image corner, in world coordinate
        center_to_corner = np.eye(4); center_to_corner[0:3,3]=center_world[0:3]

        #global transformation, including center translatioon and rigid transformation
        theorical_transformations[i,:,:] = center_to_corner @ rigid_matrix @ corner_to_center
        
        #coordinate form LR image are converted into world coordinate
        coordinate_in_world = center_to_corner @ rigid_matrix @ corner_to_center @ lr_to_world @ coordinate_in_lr
        
        #coordinate in world are converted into position in the HR image
        coordinate_in_hr = np.linalg.inv(hr_to_world) @ coordinate_in_world 
        
        #interpolate the corresponding values in HR image
        map_coordinates(image.get_fdata(),coordinate_in_hr[0:3,:],output=interpolate,order=3,mode='constant',cval=0,prefilter=False)
        new_slice = np.reshape(interpolate,(new_x,new_y,5))
        
        #sum the value along the line for the PSF
        for v in range(new_x):
            for w in range(new_y):
                low_resolution_image[v,w,i] = sum(normalised_psf*new_slice[v,w,:]) 
        
        
        #create the slice mask
        map_coordinates(mask,coordinate_in_hr[0:3,2:-1:5],output=slice_mask,order=0,mode='constant',cval=0,prefilter=False)
        new_slice = np.reshape(slice_mask,(new_x,new_y))
        for v in range(new_x):
            for w in range(new_y):
                low_resolution_mask[v,w,i] =  new_slice[v,w]
        i=i+1
        
    #create an nifti low resolution image    
    nifty_lr = Nifti1Image(low_resolution_image,lr_to_world)
    
    #create a nifi low resolution mask image
    nifty_lr_mask = Nifti1Image(low_resolution_mask,lr_to_world)
    
    return nifty_lr,nifty_lr_mask,parameters,theorical_transformations






def displaySampling(image,TransfoImages):
    """
    The function display the sampling point that are taken on the HR image, when simulated motion in a low resolution image
    image is the HR original image 
    TransfoImages is the list of transformation applied to the low resolution slice. Transfo is an array of slice three, the first element are transformation of the axial image, second from the coronal and third, from the sagittal.

    """
    
    #result image
    X,Y,Z = image.shape
    res = np.zeros((X,Y,Z))
    
    sliceRes=6
    
    imageAffine=image.affine
    
    #Transformation are considered for three stacks: axial, coronal and sagittal
    for indexTransfo in range(len(TransfoImages)):
        
        TR = TransfoImages[indexTransfo]
        
        #create low resolution images,  in axial, coronal and sagittal
        if indexTransfo==0 :
            S1=X;S2=Y;S3=Z
            transfo = np.array([[1,0,0,0],[0,1,0,0],[0,0,sliceRes,0],[0,0,0,1]])
        elif indexTransfo==1 :
            S1=X;S2=Z;S3=Y
            transfo = np.array([[1,0,0,0],[0,0,sliceRes,0],[0,1,0,0],[0,0,0,1]])
        elif indexTransfo==2 :
            S1=Z;S2=Y;S3=X #Z,Y
            transfo = np.array([[0,0,sliceRes,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
            
        LRAffine = imageAffine @ transfo
        nbTransfo = TR.shape[0]

        zi=0
        for it in range(nbTransfo):
            
            T1=TR[it,:,:]
            
            coordinate_in_lr = np.zeros((4,S1*S2*6)) #initialisation of coordinate in the low resolution image, with 6 points per voxels
            #create the coordinate of points of the slice i in the LR image, with 6 points per voxel, center in 0
            ii = np.arange(0,S1) 
            jj = np.arange(0,S2)
    
            zz = np.linspace(0,1,6,endpoint=False)
            
            iv,jv,zv = np.meshgrid(ii,jj,zz,indexing='ij')

            iv = np.reshape(iv, (-1))
            jv = np.reshape(jv, (-1))
            zv = np.reshape(zv, (-1))
            
            
            coordinate_in_lr[0,:] = iv
            coordinate_in_lr[1,:] = jv
            coordinate_in_lr[2,:] = zi+zv
            coordinate_in_lr[3,:] = 1#np.ones(S1*S2*1)
            
            coordinate_in_world = T1 @ LRAffine @ coordinate_in_lr
            coordinate_in_hr = np.round(np.linalg.inv(image.affine) @ coordinate_in_world).astype(int) #np.linalg.inv(image.affine) @ coordinate_in_world
            
            zi=zi+1
            nb_point=coordinate_in_hr[0:3,:].shape[1]
            
            for p in range(nb_point):
                x,y,z=coordinate_in_hr[0:3,p]
                if x<X  and x>0 and y>0 and y<Y and z>0 and z<Z:
                    res[x,y,z]=image.get_fdata()[x,y,z]
           
        
    img_res=Nifti1Image(res,image.affine)
    return img_res

