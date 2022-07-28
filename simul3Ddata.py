#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:49:35 2022

@author: mercier
"""

import numpy as np
import nibabel as nib
from tools import rigidMatrix
from data_simulation import create3VolumeFromAnImage
from tools import rigidMatrix, rotationCenter
from scipy.ndimage import map_coordinates
import random as rd


def psf(x_0,x): 
   
    FHWM = 3.0 #FHWM is equal to the slice thikness (for ssFSE sequence), here 3, cf article Jiang et al.
    sigma = FHWM/(2*np.log(2))
    psf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp((-(x-x_0)**2)/2*sigma**2)
    
    return psf

def extract_mask(NiftiMask):
   
    mask=NiftiMask.get_fdata()
    X,Y,Z=NiftiMask.shape
    mask=mask>0
    newMask = nib.Nifti1Image(mask.astype(int),NiftiMask.affine)
    
    return newMask

def simulateMvt(image,AngleMinMax,TransMinMax,orientation,mask=np.nan,mvt=True):
    """
    The function create 3 orthogonals volume with a 3D mri image

    Inputs : 
    image : 3D mri volume

    Returns :
    Volume: Volume in the choosen orientation


    """
    X,Y,Z = image.shape
    sliceRes = 6 #number of slice to subsample the image, in the final volume, we take only 1 over sliceRes images
        
    if orientation=='axial':
        S1=X;S2=Y;S3=Z
        transfo = np.array([[1,0,0,0],[0,1,0,0],[0,0,sliceRes,0],[0,0,0,1]]) #Rotation to obtain an axial orientation
      
    elif orientation=='coronal':
        S1=X;S2=Z;S3=Y
        transfo = np.array([[1,0,0,0],[0,0,sliceRes,0],[0,1,0,0],[0,0,0,1]])
        
    
    elif orientation=='sagittal':
        S1=Z;S2=Y;S3=X
        transfo = np.array([[0,0,sliceRes,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
   
    else :
        print('unkown orientation, choose between axial, coronal and sagittal')
        return 0
    
    s3=int(S3/sliceRes)
    imgLr = np.zeros((S1,S2,s3))
    parameters=np.zeros((s3,6))
    TransfoLR=np.zeros((s3,4,4))
    vect = np.linspace(0,s3-1,s3,dtype=int)
    print(vect)
    imageAffine = image.affine
    LRAffine = imageAffine @ transfo
    print(LRAffine)
    
    if ~np.all(np.isnan(mask)):
        newMask=np.zeros((S1,S2,s3))
    
    for i in vect: #Create the axial image
        
        if mvt==False: #if no movment, T is the identity
            T = np.eye(4)
            parameters[i,:]= np.array([0,0,0,0,0,0])
        else : #else create the movment with random parameters
            RangeAngle=AngleMinMax[1]-AngleMinMax[0]
            RangeTranslation=TransMinMax[1]-TransMinMax[0]
            a1 = rd.random()*(RangeAngle) - (RangeAngle)/2
            a2 = rd.random()*(RangeAngle) - (RangeAngle)/2
            a3 = rd.random()*(RangeAngle) - (RangeAngle)/2
            t1 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            t2 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            t3 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            T = rigidMatrix([a1,a2,a3,t1,t2,t3])
            parameters[i,:]= np.array([a1,a2,a3,t1,t2,t3])
        
        coordinate_in_lr = np.zeros((4,S1*S2*6)) #initialisation of coordinate in the low resolution image, with 6 points per voxels
        output = np.zeros(S1*S2*6) #output of the interpolation
        
        #create the coordinate of points of the slice i in the LR image, with 6 points per voxel, center in 0
        ii = np.arange(0,S1) 
        jj = np.arange(0,S2)

        zz = np.linspace(-0.45,0.45,6)
        
        iv,jv = np.meshgrid(ii,jj,indexing='ij')
        
        iv = np.reshape(iv, (-1))
        jv = np.reshape(jv, (-1))
        
        iv,zv = np.meshgrid(iv,zz,indexing='ij')
        jv,zv = np.meshgrid(jv,zz,indexing='ij')
        
        iv = np.reshape(iv, (-1))
        jv = np.reshape(jv, (-1))
        zv = np.reshape(zv, (-1))
        
        coordinate_in_lr[0,:] = iv
        coordinate_in_lr[1,:] = jv
        coordinate_in_lr[2,:] = zv + i
        coordinate_in_lr[3,:] = np.ones(S1*S2*6)
        
        #the transformation is applied at the center of the image
        if np.all(np.isnan(mask)):
            center = np.ones(4); center[0] = int(S1/2); center[1] = int(S2/2); center[2] = i; center[3]= 1
        else:
            center=rotationCenter(mask[:,:,i])
        
        center = LRAffine @ center
        
        matrix_center = np.eye(4); matrix_center[0:3,3]=-center[0:3]
        matrix_invcenter = np.eye(4); matrix_invcenter[0:3,3]=center[0:3]

        #corresponding position in the hr image
        TransfoLR[i,:,:] = matrix_invcenter @ T @ matrix_center @ LRAffine
        coordinate_in_world = matrix_invcenter @ T @ matrix_center @ LRAffine @ coordinate_in_lr
        coordinate_in_hr = np.linalg.inv(image.affine) @ coordinate_in_world
        
        #interpolate the corresponding values in HR image
        map_coordinates(image.get_fdata(),coordinate_in_hr[0:3,:],output=output,order=1,mode='constant',cval=np.nan,prefilter=False)
        new_slice = np.reshape(output,(S1,S2,6))
        
        #compute intensity value in lr image using the psf
        var=0
        for v in range(S1):
            for w in range(S2):
                imgLr[v,w,i] = sum(psf(0,zz) * new_slice[v,w,:]) / 6
                var=var+6
        
        if ~np.all(np.isnan(mask)):
            outputMask=np.zeros((S1*S2*6))
            map_coordinates(mask,coordinate_in_hr[0:3,:],output=outputMask,order=1,mode='constant',cval=np.nan,prefilter=False)
            new_slice = np.reshape(outputMask,(S1,S2,6))
            var=0
            for v in range(S1):
                for w in range(S2):
                    newMask[v,w,i] = sum(psf(0,zz) * new_slice[v,w,:]) / 6
                    var=var+6
        
        i=i+1
        
    Volume = nib.Nifti1Image(imgLr,LRAffine)
    
    if ~np.all(np.isnan(mask)):
        VolumeMask = nib.Nifti1Image(newMask,LRAffine)
        return Volume,VolumeMask,parameters,TransfoLR
    
    return Volume,parameters,TransfoLR
