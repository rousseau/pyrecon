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

#PSF is defined for the trough plan direction
def psf(x_0,x):  
   
    # #The PSF is a gaussian function
    FHWM = 1.0 #FHWM is equal to the slice thikness (for ssFSE sequence), 1 in voxel  (cf article Jiang et al.)
    sigma = FHWM/(2*np.sqrt((2*np.log(2))))
    res = (1/(sigma*np.sqrt(2*np.pi)))* np.exp((-(x-x_0)**2)/(2*(sigma**2)))

    return res

def extract_mask(NiftiMask):
   """
    Create a binary mask of the brain from a brain segmentation image

   """
   mask=NiftiMask.get_fdata()
   X,Y,Z=NiftiMask.shape
   mask[mask==4]=0
   mask=mask>1e-2
   newMask = nib.Nifti1Image(mask.astype(np.float),NiftiMask.affine)
    
   return newMask

def simulateMvt(image,AngleMinMax,TransMinMax,resolution,orientation,mask,mvt=True):
    """
    The function create 3 orthogonals low resolution volume with from 3D mri image

    Inputs : 
    image : 3D mri volume

    Returns :
    Volume: Volume in the choosen orientation


    """
    X,Y,Z = image.shape
    sliceRes = resolution #number of slice to subsample the image, in the final volume, we take only 1 over sliceRes images (in our application, we choose 6 to have a final image of resolution 0.5x0.5x1)

    
    #create an LR image in axial orientation
    if orientation=='axial':
        #S3 coordinate must correspond to the througt plan direction
        S1=X;S2=Y;S3=Z
        #modify image orientation
        transfo = np.array([[1,0,0,0],[0,1,0,0],[0,0,sliceRes,0],[0,0,0,1]]) 

        
    #create an LR image in coronal orientation
    elif orientation=='coronal':
        #S3 coordinate must correspond to the througt plan direction
        S1=X;S2=Z;S3=Y
        #modify image orientation
        transfo = np.array([[1,0,0,0],[0,0,sliceRes,0],[0,1,0,0],[0,0,0,1]])

        
        
    #create an LR image in sagittal orientation
    elif orientation=='sagittal':
        #S3 coordinate must correspond to the througt plan direction
        S1=Z;S2=Y;S3=X 
        #modify image orientation
        transfo = np.array([[0,0,sliceRes,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])

   
    else :
        print('unkown orientation, choose between axial, coronal and sagittal')
        return 0
    
    #number of slices
    s3=S3//sliceRes
    
    #create an LR image
    imgLr = np.zeros((S1,S2,s3))
    
    #creta an LR mask
    newMask = np.zeros((S1,S2,s3))
    
    #6 parameters, for each slices
    parameters=np.zeros((s3,6))
    
    #a rigid matrix in homogenous coordinate for each slice
    TransfoLR=np.zeros((s3,4,4))
    
    #slice number
    vect = np.linspace(0,s3-1,s3,dtype=int)
    
    #coordinate to world matrix of the HR image
    imageAffine = image.affine
    
    #coordinate to worl matrix of the LR image
    LRAffine =  imageAffine @ transfo

    
    #point coordinate for the PSF (in voxel coordinate)
    zz = np.linspace(-0.5,0.5,5)
    
    #PSF evaluated on zz points 
    PSF = psf(0,zz) 
    
    #normalized PSF on zz points
    normPSF = PSF/sum(PSF)
    
    #For each slice of the low resolution image :
        #1/ Simulate rigid motion if we want some. If not, motion matrix is the identity
        #2/ Interpolate corresponding values from the HR image 
        #3/ Interpolate corresponding values for the HR mask image
        
    for i in vect: 
        
        #No motion, motion matrix is the identity 
        if mvt==False: 
            T = np.eye(4)
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
            T = rigidMatrix([a1,a2,a3,t1,t2,t3])
            parameters[i,:]= np.array([a1,a2,a3,t1,t2,t3])
        
        #coordinate in the low resolution image, in homogenous coordinate. 
        coordinate_in_lr = np.zeros((4,S1*S2*5)) 
        
        #output of the interpolation
        output = np.zeros(S1*S2*5) 
        
        #LR mask
        outputMask=np.zeros((S1*S2))
        
        #coordinate, in voxel, of the slice i in the LR image
        ii = np.arange(0,S1) 
        jj = np.arange(0,S2)

        
        iv,jv, zv = np.meshgrid(ii,jj,zz,indexing='ij')
        
        
        iv = np.reshape(iv, (-1))
        jv = np.reshape(jv, (-1))
        zv = np.reshape(zv, (-1))
        
        coordinate_in_lr[0,:] = iv
        coordinate_in_lr[1,:] = jv
        coordinate_in_lr[2,:] = i+zv
        coordinate_in_lr[3,:] = 1
        
        #center coordinate, of the slice, in image coordinate
        center = np.ones(4); center[0] = S1//2; center[1] = S2//2; center[2] = i; center[3]= 1
        
        #center coordinate, in voxel, in world coordinate
        center = LRAffine @ center
        
        #translation matrix, from image corner to image center, in world coordinate
        matrix_center = np.eye(4); matrix_center[0:3,3]=-center[0:3]
        
        #translation matrix, from image center to image corner, in world coordinate
        matrix_invcenter = np.eye(4); matrix_invcenter[0:3,3]=center[0:3]

        #global transformation, including center translatioon and rigid transformation
        TransfoLR[i,:,:] = matrix_invcenter @ T @ matrix_center
        
        #coordinate form LR image are converted into world coordinate
        coordinate_in_world = matrix_invcenter @ T @ matrix_center @ LRAffine @ coordinate_in_lr
        
        #coordinate in world are converted into position in the HR image
        coordinate_in_hr = np.linalg.inv(image.affine) @ coordinate_in_world 
        
        #interpolate the corresponding values in HR image
        map_coordinates(image.get_fdata(),coordinate_in_hr[0:3,:],output=output,order=3,mode='constant',cval=0,prefilter=False)
        new_slice = np.reshape(output,(S1,S2,5))
        
        #sum the value along the line for the PSF
        for v in range(S1):
            for w in range(S2):
                imgLr[v,w,i] = sum(normPSF*new_slice[v,w,:]) #new_slice[v,w,:] 
        
        
        #create a LR mask
        map_coordinates(mask,coordinate_in_hr[0:3,2:-1:5],output=outputMask,order=0,mode='constant',cval=0,prefilter=False)
        new_slice = np.reshape(outputMask,(S1,S2))
        for v in range(S1):
            for w in range(S2):
                newMask[v,w,i] =  new_slice[v,w]
        i=i+1
        
    #create an nifti low resolution image    
    Volume = nib.Nifti1Image(imgLr,LRAffine)
    
    #create a nifi low resolution mask image
    VolumeMask = nib.Nifti1Image(newMask,LRAffine)
    
    return Volume,VolumeMask,parameters,TransfoLR


def findCommonPointbtw2V(Volume1,Volume2,rejectedSlices):
    """
    The function take as parameters, 2 list of slices, corresponding to a simulate LR image without motion and compute identical point on the two images. Those points are selected to compute the error of registration.
    
    rejectedSlice correspond to slices that have not been well register with the registration algorithm, they are not taken in volume1 and volume2 
    
    """
    
    #list of points in volume 1
    listPointVolume1 = [] 
    
    #list of points in volume 2
    listPointVolume2 = []
    
    #For each slice in volume1
    for zV1 in range(len(Volume1)):
        
        sV1 = Volume1[zV1]
        maskV1 = sV1.get_mask()
        
        #For each slice in volume 2
        for zV2 in range(len(Volume2)):
            sV2 = Volume2[zV2]
            maskV2 = sV2.get_mask()
            
            #We check if slices sV1 and sV2 are rejected or not, if they are not rejected, we continue
            if ((sV1.get_orientation(),sV1.get_index_slice()) not in rejectedSlices) and ((sV2.get_orientation(),sV2.get_index_slice()) not in rejectedSlices) :
                
                sliceimage1=sV1.get_slice().get_fdata();M1=sV1.get_transfo();res=min(sV1.get_slice().header.get_zooms())
                sliceimage2=sV2.get_slice().get_fdata();M2=sV2.get_transfo()
                
                #compute common segment between sV1 and sV2
                pt1,pt2,nbpoint,ok = commonSegment(sliceimage1,M1,sliceimage2,M2,res)
                #nbpoint is the number of points we consider on the intersection and ok is a boolean wich indicate if an intersection line has been computed
                #they are in array because commonSegment used numba and result must be the same type
                nbpoint=np.int32(nbpoint[0,0]);ok=np.int32(ok[0,0]) 
                nbpt=nbpoint
                
                #list of point coordinate in V1
                pointV1 = np.zeros((nbpt,3))
                pointV1[:,0] = np.linspace(pt1[0,0],pt1[0,1],nbpt)
                pointV1[:,1] = np.linspace(pt1[1,0],pt1[1,1],nbpt)
                pointV1[:,2] = np.ones(nbpt)*zV1
                interpolMaskV1 = np.zeros(nbpt)
                ptInterpol = np.zeros((3,nbpt))
                ptInterpol[0,:] = pointV1[:,0]
                ptInterpol[1,:] = pointV1[:,1]
                ptInterpol[2,:] = np.zeros(nbpt)
                
                
                #list of point coordinate in V2
                pointV2 = np.zeros((nbpt,3))
                pointV2[:,0] = np.linspace(pt2[0,0],pt2[0,1],nbpt)
                pointV2[:,1] = np.linspace(pt2[1,0],pt2[1,1],nbpt)
                pointV2[:,2] = np.ones(nbpt)*zV2
                interpolMaskV2 = np.zeros(nbpt)
                ptInterpol = np.zeros((3,nbpt))
                ptInterpol[0,:] = pointV2[:,0]
                ptInterpol[1,:] = pointV2[:,1]
                ptInterpol[2,:] = np.zeros(nbpt)
                
                
                #interpolation of the coordinate on the mask, in volume1. In the final coordinate, we want to take only values that are on the mask
                map_coordinates(maskV1, ptInterpol, output=interpolMaskV1, order=0, mode='constant',cval=np.nan,prefilter=False)
                
                
                #interpolation of the coordinate on the mask, in volume2. In the final coordinate, we want to take only values that are on the mask
                map_coordinates(maskV2, ptInterpol, output=interpolMaskV2, order=0, mode='constant',cval=0,prefilter=False)
                
                #we choose values that are on the mask in volume 1 and 2
                pV1 = pointV1[np.where(interpolMaskV1>0) or np.where(interpolMaskV2>0)]
                listpoint = pV1.tolist()
                listPointVolume1.extend(listpoint)
                
                pV2 = pointV2[np.where(interpolMaskV1>0) or np.where(interpolMaskV2>0)]
                listpoint = pV2.tolist()
                listPointVolume2.extend(listpoint)
            
            
            
     #debug :
     #The debug procedure check if the coordinate pV1 in volume1 and pV2 in volume2 are identiqual in world coordinate
    
    # for indice  in range(len(listPointVolume1)):
            
    #         p1 = listPointVolume1[indice]
    #         p2 = listPointVolume2[indice]
            
    #         zV1 = int(p1[2])
    #         zV2 = int(p2[2])
            
    #         transV1 = Volume1[zV1].get_transfo()
    #         transV2 = Volume2[zV2].get_transfo() 
    #         pwv1 = np.zeros(4)
    #         pwv1[0:2] = p1[0:2]
    #         pwv1[3] = 1
    #         pwv2 = np.zeros(4)
    #         pwv2[0:2] = p2[0:2]
    #         pwv2[3] = 1
    #         pv1_inw = transV1 @ pwv1
    #         pv2_inw = transV2 @ pwv2
                     
    #         if ~((pv1_inw==pv2_inw).all()) :
    #             print("error : Points are not the same")
    #             break
            
    return np.array(listPointVolume1),np.array(listPointVolume2)

def ChamferDistance(volume):
    """
    Compute the chamfer distance map with the border of the mask
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
    
    """
    The function ErrorOfRegistrationBtw2Slice compute the distance between  2 list of points in the corrected motion simulated volume
    Volume1 : list of 2D slices from 1 image, with corrected motion
    Volume2 : list of 2D slices, from an other image, with corrected motion
    pointCoordinateV1 : list of points in volume1 on wich we compute the distance
    pointCoordinateV2 : list of points in volume2 on wich we compute the distance
    transfo1 : list of  inverse simulated transfo for each slices #it includes : M-c @ T @ Mc with Mc and M-c translation from image to center and from center to image, and T the rigid simulated transformation
    transfo2 : same but for the second volume
    """
    
    
    nbpointError = pointCoordinateV1.shape[0]
    res = np.zeros(nbpointError)
    
    #the distance is compute between each points of pointCoordinateV1 and pointCoordinateV2
    for ip in range(nbpointError):
        
        #coordinate of the point in volume1
        pV1 = pointCoordinateV1[ip,:]
        
        #we need the z coordinate to know the index of the slice
        zV1 = int(pV1[2])

        #postion of the point after simulation and correction of the motion
        #transformation global : the inverse simulated transformation is applied to the corrected transformation
        Mv1 = transfo1[Volume1[zV1].get_index_slice(),:,:]  @ Volume1[zV1].get_transfo()
        p2Dv1 = np.zeros(4)
        p2Dv1[0:2] = pV1[0:2]
        p2Dv1[3] = 1
        
        #position of the point, in world coordinate, after application and correction of the motion
        pInV1 = Mv1 @ p2Dv1
        
        #coordinate of the point in volume2
        pV2 = pointCoordinateV2[ip,:]
        
        #we need the z coordinate to know the index of the slice
        zV2 = int(pV2[2])
        
        #position of the point after simulation and correction of the motion
        #transformation global : the inverse simulated transformation is applied to the corrected transformation
        Mv2 = transfo2[Volume2[zV2].get_index_slice(),:,:] @ Volume2[zV2].get_transfo()
        p2Dv2 = np.zeros(4)
        p2Dv2[0:2] = pV2[0:2]
        p2Dv2[3] = 1
        
        #position of the point, in world coordinate, after application and correction of the motion
        pInV2 = Mv2 @ p2Dv2
        
        #distance between the two points in world coordinate system
        diff = np.sqrt((pInV1[0]-pInV2[0])**2 + (pInV1[1]-pInV2[1])**2 + (pInV1[2]-pInV2[2])**2)
        res[ip]=diff
      
    return res


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
           
        
    img_res=nib.Nifti1Image(res,image.affine)
    return img_res

