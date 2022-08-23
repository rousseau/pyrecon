#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:49:35 2022

@author: mercier
"""

import numpy as np
import nibabel as nib
from tools import rigidMatrix
from scipy.ndimage import map_coordinates
import random as rd
from registration import commonSegment
from scipy.ndimage import distance_transform_cdt

def psf(x_0,x): 
   
    FHWM = 3.0 #FHWM is equal to the slice thikness (for ssFSE sequence), here 3, cf article Jiang et al.
    sigma = FHWM/(2*np.log(2))
    psf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp((-(x-x_0)**2)/2*sigma**2)
    
    return psf

def extract_mask(NiftiMask):
   """
    Create a binary mask of the brain from a brain segmentation image

   """
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
    
    zz = np.linspace(-1,1,21)
    PSF = psf(0,zz) 
    normPSF = PSF/sum(PSF)
    
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
        
        coordinate_in_lr = np.zeros((4,S1*S2*21)) #initialisation of coordinate in the low resolution image, with 6 points per voxels
        output = np.zeros(S1*S2*21) #output of the interpolation
        
        #create the coordinate of points of the slice i in the LR image, with 6 points per voxel, center in 0
        ii = np.arange(0,S1) 
        jj = np.arange(0,S2)

        
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
        coordinate_in_lr[3,:] = np.ones(S1*S2*21)
        
        #the transformation is applied at the center of the image
        center = np.ones(4); center[0] = int(S1/2); center[1] = int(S2/2); center[2] = i; center[3]= 1
        
        center = LRAffine @ center
        
        matrix_center = np.eye(4); matrix_center[0:3,3]=-center[0:3]
        matrix_invcenter = np.eye(4); matrix_invcenter[0:3,3]=center[0:3]

        #corresponding position in the hr image
        TransfoLR[i,:,:] = matrix_invcenter @ T @ matrix_center 
        coordinate_in_world = matrix_invcenter @ T @ matrix_center @ LRAffine @ coordinate_in_lr
        coordinate_in_hr = np.linalg.inv(image.affine) @ coordinate_in_world
        
        #interpolate the corresponding values in HR image
        map_coordinates(image.get_fdata(),coordinate_in_hr[0:3,:],output=output,order=3,mode='constant',cval=np.nan,prefilter=False)
        new_slice = np.reshape(output,(S1,S2,21))
        
        #compute intensity value in lr image using the psf
        var=0
        for v in range(S1):
            for w in range(S2):
                imgLr[v,w,i] = sum(normPSF*new_slice[v,w,:])
                var=var+21
        
        if ~np.all(np.isnan(mask)):
            outputMask=np.zeros((S1*S2))
            map_coordinates(mask,coordinate_in_hr[0:3,10:-1:21],output=outputMask,order=0,mode='constant',cval=np.nan,prefilter=False)
            new_slice = np.reshape(outputMask,(S1,S2))
            for v in range(S1):
                for w in range(S2):
                    newMask[v,w,i] =  new_slice[v,w]

        
        i=i+1
        
    Volume = nib.Nifti1Image(imgLr,LRAffine)
    
    if ~np.all(np.isnan(mask)):
        VolumeMask = nib.Nifti1Image(newMask,LRAffine)
        return Volume,VolumeMask,parameters,TransfoLR
    
    return Volume,parameters,TransfoLR


def findCommonPointbtw2V(Volume1,Volume2,rejectedSlices):
    
    listPointVolume1 = []
    listPointVolume2 = []
    
    for zV1 in range(len(Volume1)):
        sV1 = Volume1[zV1]
        maskV1 = sV1.get_mask()
        for zV2 in range(len(Volume2)):
            sV2 = Volume2[zV2]
            maskV2 = sV2.get_mask()
            
            print((sV1.get_orientation(),sV1.get_index_slice()))
            if ((sV1.get_orientation(),sV1.get_index_slice()) not in rejectedSlices) and ((sV2.get_orientation(),sV2.get_index_slice()) not in rejectedSlices) :
                print('please work')
                sliceimage1=sV1.get_slice().get_fdata();M1=sV1.get_transfo();res=min(sV1.get_slice().header.get_zooms())
                sliceimage2=sV2.get_slice().get_fdata();M2=sV2.get_transfo()
                pt1,pt2,nbpoint,ok = commonSegment(sliceimage1,M1,sliceimage2,M2,res)
                nbpoint=np.int32(nbpoint[0,0]);ok=np.int32(ok[0,0])
                dist = 1
                nbpt = int(np.ceil(nbpoint/dist))
    
                pointV1 = np.zeros((nbpt,3))
                pointV1[:,0] = np.linspace(pt1[0,0],pt1[0,1],nbpt)
                pointV1[:,1] = np.linspace(pt1[1,0],pt1[1,1],nbpt)
                pointV1[:,2] = np.ones(nbpt)*zV1
                interpolMaskV1 = np.zeros(nbpt)
                ptInterpol = np.zeros((3,nbpt))
                ptInterpol[0,:] = pointV1[:,0]
                ptInterpol[1,:] = pointV1[:,1]
                ptInterpol[2,:] = np.zeros(nbpt)
                
                
                
                pointV2 = np.zeros((nbpt,3))
                pointV2[:,0] = np.linspace(pt2[0,0],pt2[0,1],nbpt)
                pointV2[:,1] = np.linspace(pt2[1,0],pt2[1,1],nbpt)
                pointV2[:,2] = np.ones(nbpt)*zV2
                interpolMaskV2 = np.zeros(nbpt)
                ptInterpol = np.zeros((3,nbpt))
                ptInterpol[0,:] = pointV2[:,0]
                ptInterpol[1,:] = pointV2[:,1]
                ptInterpol[2,:] = np.zeros(nbpt)
                
                
                map_coordinates(maskV1, ptInterpol, output=interpolMaskV1, order=0, mode='constant',cval=np.nan,prefilter=False)
                
                
                
                map_coordinates(maskV2, ptInterpol, output=interpolMaskV2, order=0, mode='constant',cval=np.nan,prefilter=False)
                
                pV1 = pointV1[np.where(interpolMaskV1>0) or np.where(interpolMaskV2>0)]
                listpoint = pV1.tolist()
                listPointVolume1.extend(listpoint)
                
                pV2 = pointV2[np.where(interpolMaskV1>0) or np.where(interpolMaskV2>0)]
                listpoint = pV2.tolist()
                listPointVolume2.extend(listpoint)
            
            
            
     #debug : 
    
    for indice  in range(len(listPointVolume1)):
            
            #print('indice :', indice)
            p1 = listPointVolume1[indice]
            p2 = listPointVolume2[indice]
            
            zV1 = int(p1[2])
            zV2 = int(p2[2])
            
            transV1 = Volume1[zV1].get_transfo()
            transV2 = Volume2[zV2].get_transfo() 
            pwv1 = np.zeros(4)
            pwv1[0:2] = p1[0:2]
            pwv1[3] = 1
            pwv2 = np.zeros(4)
            pwv2[0:2] = p2[0:2]
            pwv2[3] = 1
            pv1_inw = transV1 @ pwv1
            pv2_inw = transV2 @ pwv2
            #print('1 :', pv1_inw)
            #print('2 :', pv2_inw)
                     
            if ~((pv1_inw==pv2_inw).all()) :
                print("error : Points are not the same")
                break
            
    return np.array(listPointVolume1),np.array(listPointVolume2)

def ChamferDistance(volume):
    """
    Compute the chamfer distance map the border of the mask
    """
    
    inv_chamfer_distance = distance_transform_cdt(volume.get_fdata())

    return inv_chamfer_distance

def createArrayOfChamferDistance(ChamferDistance,coordinateInV):
    """
    give the chamfer distance for each point in coordinateInV
    """
    
    dFromCenter = np.zeros(coordinateInV.shape[0])
    indice = 0
    
    for c in coordinateInV:    
        dFromCenter[indice] = ChamferDistance[int(c[0]),int(c[1]),int(c[2])]
        indice = indice + 1
    return dFromCenter

    
def ErrorOfRegistrationBtw2Slice(pointCoordinateV1,pointCoordinateV2,Volume1,Volume2,transfo1,transfo2):
    
    nbpointError = pointCoordinateV1.shape[0]
    res = np.zeros(nbpointError)
    
    for ip in range(nbpointError):
        
        pV1 = pointCoordinateV1[ip,:]
        
        zV1 = int(pV1[2])


        Mv1 = transfo1[Volume1[zV1].get_index_slice(),:,:]  @ Volume1[zV1].get_transfo()
        p2Dv1 = np.zeros(4)
        p2Dv1[0:2] = pV1[0:2]
        p2Dv1[3] = 1
        pInV1 = Mv1 @ p2Dv1
        
        pV2 = pointCoordinateV2[ip,:]
        
        zV2 = int(pV2[2])

        
        Mv2 = transfo2[Volume2[zV2].get_index_slice(),:,:] @ Volume2[zV2].get_transfo()
        p2Dv2 = np.zeros(4)
        p2Dv2[0:2] = pV2[0:2]
        p2Dv2[3] = 1
        pInV2 = Mv2 @ p2Dv2
        
        #print(pInV1[2],pInV2[2])
        diff = np.sqrt((pInV1[0]-pInV2[0])**2 + (pInV1[1]-pInV2[1])**2 + (pInV1[2]-pInV2[2])**2)
        res[ip]=diff
      
    return res




        
                
                
                

    
    