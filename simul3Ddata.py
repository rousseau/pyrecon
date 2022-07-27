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
from tools import rigidMatrix
from scipy.ndimage import map_coordinates
import random as rd



def psf(x_0,x): 
   
    FHWM = 3.0 #FHWM is equal to the slice thikness (for ssFSE sequence), here 3, cf article Jiang et al.
    sigma = FHWM/(2*np.log(2))
    psf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp((-(x-x_0)**2)/2*sigma**2)
    
    return psf

def create3VolumeFromAnImage(image,RangeAngle,RangeTranslation,mvt):
    """
    The function create 3 orthogonals volume with a 3D mri image

    Inputs : 
    image : 3D mri volume

    Returns :
    Volumeaxial : Volume in axial orientation
    VolumeCoronal : Volume in sagittal orientation
    VolumeSagittal : Volume in corronal orientation

    """
    X,Y,Z = image.shape
    sliceRes = 3 #number of slice to subsample the image, in the final volume, we take only 1 over sliceRes images
    z = int(Z/sliceRes)
    y = int(Y/sliceRes)
    x = int(X/sliceRes)
    img_axial = np.zeros((X,Y,z))
    img_coronal = np.zeros((X,Z,y))
    img_sagittal = np.zeros((Z,Y,x))
    vectz = np.linspace(0,Z-1,z,dtype=int)
    vecty = np.linspace(0,Y-1,y,dtype=int)
    vectx = np.linspace(0,X-1,x,dtype=int)
    
    
    i=0
    imageAffine = image.affine
    transfoAx = np.array([[1,0,0,0],[0,1,0,0],[0,0,sliceRes,0],[0,0,0,1]]) #Rotation to obtain an axial orientation
    axAffine = imageAffine @ transfoAx
    
    parametersAx=np.zeros(z,6)
    parametersCor=np.zeros(y,6)
    parametersSag=np.zeros(x,6)
    
    for index in vectz: #Create the axial image
        
        if mvt==False: #if no movment, T is the identity
            T = np.eye(4)
            parametersAx[i]= np.array([0,0,0,0,0,0])
        else : #else create the movment with random parameters
            a1 = rd.random()*(RangeAngle) - (RangeAngle)/2
            a2 = rd.random()*(RangeAngle) - (RangeAngle)/2
            a3 = rd.random()*(RangeAngle) - (RangeAngle)/2
            t1 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            t2 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            t3 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            T = rigidMatrix([a1,a2,a3,t1,t2,t3])
            parametersAx[i]= np.array([a1,a2,a3,t1,t2,t3])
        
        coordinate_in_lr = np.zeros((4,X*Y*6)) #initialisation of coordinate in the low resolution image, with 6 points per voxels
        output = np.zeros(X*Y*6) #output of the interpolation
        
        #create the coordinate of points of the slice i in the LR image, with 6 points per voxel, center in 0
        ii = np.arange(0,X) 
        jj = np.arange(0,Y)

        zz = np.linspace(-0.45,0.4,6)
        
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
        coordinate_in_lr[3,:] = np.ones(X*Y*6)
        
        #the transformation is applied at the center of the image
        center = np.ones(4); center[0] = int(X/2); center[1] = int(Y/2); center[2] = i; center[3]= 1
        center = axAffine @ center
        matrix_center = np.eye(4); matrix_center[0:3,3]=-center[0:3]
        matrix_invcenter = np.eye(4); matrix_invcenter[0:3,3]=center[0:3]

        #corresponding position in the hr image
        coordinate_in_world = matrix_invcenter @ T @ matrix_center @ axAffine @ coordinate_in_lr
        coordinate_in_hr = np.linalg.inv(image.affine) @ coordinate_in_world
        
        #interpolate the corresponding values in HR image
        map_coordinates(image.get_fdata(),coordinate_in_hr[0:3,:],output=output,order=1,mode='constant',cval=np.nan,prefilter=False)
        new_slice = np.reshape(output,(X,Y,6))

        #compute intensity value in lr image using the psf
        var=0
        for v in range(X):
            for w in range(Y):
                img_axial[v,w,i] = sum(psf(0,zz) * new_slice[v,w,:]) / 6
                var=var+6
        
        i=i+1
        
    
    
    transfoCor = np.array([[1,0,0,0],[0,0,sliceRes,0],[0,1,0,0],[0,0,0,1]]) #Rotation to obtain a coronal orientation
    corAffine =  imageAffine @ transfoCor
    
    i=0
    for index in vecty: #create the coronal image
    
        if mvt==False: #if no movment, T is the identity
            T = np.eye(4)
            parametersCor[i]= np.array([0,0,0,0,0,0])
        else : #else create the movment with random parameters
            a1 = rd.random()*(RangeAngle) - (RangeAngle)/2
            a2 = rd.random()*(RangeAngle) - (RangeAngle)/2
            a3 = rd.random()*(RangeAngle) - (RangeAngle)/2
            t1 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            t2 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            t3 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            T = rigidMatrix([a1,a2,a3,t1,t2,t3])
            parametersCor[i]= np.array([a1,a2,a3,t1,t2,t3])
        

        coordinate_in_lr = np.zeros((4,X*Z*6))
        output = np.zeros(X*Z*6)
        
        ii = np.arange(0,X)
        jj = np.arange(0,Z)
        
        zz = np.linspace(-0.45,0.4,6)

        
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
        coordinate_in_lr[3,:] = np.ones(X*Z*6)

        center = np.ones(4); center[0] = int(X/2); center[1] = int(Z/2); center[2] = i; center[3]= 1
        center = axAffine @ center
        matrix_center = np.eye(4); matrix_center[0:3,3]=-center[0:3]
        matrix_invcenter = np.eye(4); matrix_invcenter[0:3,3]=center[0:3]

        coordinate_in_world = matrix_invcenter @ T @ matrix_center @ corAffine @ coordinate_in_lr
        coordinate_in_hr = np.linalg.inv(image.affine) @ coordinate_in_world

        
        map_coordinates(image.get_fdata(),coordinate_in_hr[0:3,:],output=output,order=1,mode='constant',cval=np.nan,prefilter=False)
        
        new_slice = np.reshape(output,(X,Z,6))


        var=0
        for v in range(X):
            for w in range(Z):
                img_coronal[v,w,i] = sum(psf(0,zz) * new_slice[v,w,:]) / 6
                var=var+6

        i=i+1

    transfoSag = np.array([[0,0,sliceRes,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]) #Rotation to obtain a sagittal orientation
    sagAffine =   imageAffine @ transfoSag

    i=0
    for index in vectx: #create the sagital image
    
        if mvt==False: #if no movment, T is the identity
            T = np.eye(4)
            parametersSag[i]= np.array([0,0,0,0,0,0])
        else : #else create the movment with random parameters
            a1 = rd.random()*(RangeAngle) - (RangeAngle)/2
            a2 = rd.random()*(RangeAngle) - (RangeAngle)/2
            a3 = rd.random()*(RangeAngle) - (RangeAngle)/2
            t1 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            t2 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            t3 = rd.random()*(RangeTranslation) - (RangeTranslation)/2
            T = rigidMatrix([a1,a2,a3,t1,t2,t3])
            parametersSag[i]= np.array([a1,a2,a3,t1,t2,t3])


        coordinate_in_lr = np.zeros((4,Z*Y*6))
        output = np.zeros(Z*Y*6)
        
        ii = np.arange(0,Z)
        jj = np.arange(0,Y)
        
        zz = np.linspace(-0.45,0.4,6)
        
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
        coordinate_in_lr[3,:] = np.ones(Z*Y*6)
        
        center = np.ones(4); center[0] = int(Z/2); center[1] = int(Y/2); center[2] = i; center[3]= 1
        center = axAffine @ center
        matrix_center = np.eye(4); matrix_center[0:3,3]=-center[0:3]
        matrix_invcenter = np.eye(4); matrix_invcenter[0:3,3]=center[0:3]

        coordinate_in_world = matrix_invcenter @ T @ matrix_center @ sagAffine @ coordinate_in_lr
        coordinate_in_hr = np.linalg.inv(image.affine) @ coordinate_in_world
        
        map_coordinates(image.get_fdata(),coordinate_in_hr[0:3,:],output=output,order=1,mode='constant',cval=np.nan,prefilter=False)
        new_slice = np.reshape(output,(Z,Y,6))
        
        var=0
        for v in range(Z):
            for w in range(Y):
                img_sagittal[v,w,i] = sum(psf(0,zz) * new_slice[v,w,:]) / 6
                var=var+6

        i=i+1

    Volumeaxial = nib.Nifti1Image(img_axial,  axAffine)
    VolumeCoronal = nib.Nifti1Image(img_coronal, corAffine)
    VolumeSagittal = nib.Nifti1Image(img_sagittal, sagAffine)
    
    return Volumeaxial,parametersAx,VolumeCoronal,parametersCor,VolumeSagittal,parametersSag 



#test des fonctions : 
    
#Image Isotrope : 
HRnifti = nib.load('/home/mercier/Documents/donnee/simulation/dhcp_T2w_iso1mm.nii.gz')
HRimage = HRnifti.get_fdata()

LrAxNifti,LrCorNifti,LrSagNifti = create3VolumeFromAnImage(HRnifti)

nib.save(LrAxNifti,'LrAxNifti.nii.gz')

nib.save(LrCorNifti,'LrCorNifti.nii.gz')

nib.save(LrSagNifti,'LrSagNifti.nii.gz')
