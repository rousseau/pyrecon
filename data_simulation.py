#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:38:29 2022

@author: mercier

This script aims to create motion simulation on an MRI image to validate the registration algorithm

"""

import nibabel as nib
import numpy as np
import random as rd
from kim_cm import loadSlice,commonSegment

def create3VolumeFromAlist(listSlice):
    
    SliceAx = []
    SliceCor = []
    SliceSag = []
    
    for s in listSlice:
        if s.get_orientation()=='Axial':
           SliceAx.append(s)
        elif s.get_orientation()=='Coronal':
           SliceCor.append(s)
        elif s.get_orientation()=='Sagittal':
           SliceSag.append(s)
        else:
            print('error : image orientation must be either Axial, Coronal or Sagittal')
    return SliceAx,SliceCor,SliceSag
    
def findCommonPointBetweenAxSagCor(SliceAx,SliceCor,SliceSag):
    
    listPointAxCor = []
    listPointCor = []
    listpoint = []
    
    finalListAx = []
    finalListCor = []
    finalListSag = []
    
    
    for zax in range(len(SliceAx)):
        sax = SliceAx[zax]
        
        for zcor in range(len(SliceCor)):
        #check the point on the intersection between slices axial and coronal    
            scor = SliceCor[zcor]
            pt1,pt2,nbpoint,ok = commonSegment(sax,scor)
            pointAx = np.zeros((nbpoint,3))
            pointAx[:,0] = np.linspace(pt1[0,0],pt1[0,1],nbpoint)
            pointAx[:,1] = np.linspace(pt1[1,0],pt1[1,1],nbpoint)
            pointAx[:,2] = np.ones(nbpoint)*zax
            listpoint = pointAx.tolist()
            listPointAxCor.extend(listpoint)
            # print('zax:',zax)
            # print('zcor:',zcor)
            # print('1 :',pt1[1,0],pt1[1,1])
            # print('2 : ',pt2[1,0],pt2[1,1])
            #print("pointAx:",pointAx)
            
            pointCor = np.zeros((nbpoint,3))
            pointCor[:,0] = np.linspace(pt2[0,0],pt2[0,1],nbpoint)
            pointCor[:,1] = np.linspace(pt2[1,0],pt2[1,1],nbpoint)
            pointCor[:,2] = np.ones(nbpoint)*zcor
            listpoint = pointCor.tolist()
            listPointCor.extend(listpoint)
            #print("pointCor:",pointCor)
    #print(arrayAxCor.shape)
    
    for zax in range(len(SliceAx)):  
        sax = SliceAx[zax]
        
        for zsag in range(len(SliceSag)):
        #check the point on the intersection between slices axial and sagiattal
        
            ssag = SliceSag[zsag]
            pt1,pt2,nbpoint,ok = commonSegment(sax,ssag)
            pointAx = np.zeros((nbpoint,3))
            pointAx[:,0] = np.linspace(pt1[0,0],pt1[0,1],nbpoint)
            pointAx[:,1] = np.linspace(pt1[1,0],pt1[1,1],nbpoint)
            pointAx[:,2] = np.ones(nbpoint)*zax   
            # print('zax:',zax)
            # print('zsag:',zsag)
            # print('1 :',pt1[0,0],pt1[0,1])
            # print('2 : ',pt2[0,0],pt2[0,1])
            
            
            pointSag = np.zeros((nbpoint,3))
            pointSag[:,0] = np.linspace(pt2[0,0],pt2[0,1],nbpoint)
            pointSag[:,1] = np.linspace(pt2[1,0],pt2[1,1],nbpoint)
            pointSag[:,2] = np.ones(nbpoint)*zsag
            
            
            for ip in range(pointAx.shape[0]):

                p = pointAx[ip,:]
                pl = p.tolist()
                #print(len(listPointAxCor))
                if  pl in listPointAxCor:
 
                    finalListAx.append(pl)
                
                    psag = pointSag[ip,:]
                    finalListSag.append(psag)
                    
                    ipcor = listPointAxCor.index(pl)
                    pcor = listPointCor[ipcor]
                    finalListCor.append(pcor)
                    #print(listPointAxCor[ipcor] == pl)
                    # zcor = int(pcor[2])
                    # zax = int(pl[2])
                    # zsag = int(psag[2])
                    # transCor = SliceCor[zcor].get_transfo()
                    # transSag = SliceSag[zsag].get_transfo() 
                    # transAx = SliceAx[zax].get_transfo()
                    # pwax = np.zeros(4)
                    # pwax[0:2] = pl[0:2]
                    # pwax[3] = 1
                    # pwcor = np.zeros(4)
                    # pwcor[0:2] = pcor[0:2]
                    # pwcor[3] = 1
                    # pwsag = np.zeros(4)
                    # pwsag[3] = 1
                    # pwsag[0:2] = psag[0:2]
                    # print('pax',pl)
                    # print('pcor',pcor)
                    # print('psag',psag)
                    
                    # print('pax_tow',transAx @ pwax)
                    # print('pcor_tow',transCor @ pwcor)
                    # print('psag_tow',transSag @ pwsag)
                    #pcor = listPointCor[np.where(arrayAxCor == np.array([p]))]
                    #finalListCor.append(pcor)
    
     
    pointCoordinateInAx = np.array(finalListAx) 
    pointCoordinateInCor = np.array(finalListCor)
    pointCoordinateInSag = np.array(finalListSag)
 
    # nbPointAxCor = arrayAxCor.shape[0]
    # nbPointAxSag = arrayAxSag.shape[0]

    
    # for ipAxCor in range(nbPointAxCor):
    # #check the points that intersect an axial, coronal and sagittal slice.
    #     #print(p)
       
    #     pAxCor=arrayAxCor[ipAxCor,:]
    #     print(ipAxCor)
    #     for ipAxSag in range(nbPointAxSag):
            
    #         pAxSag = arrayAxSag[ipAxSag,:]
            
    #         if (pAxCor == pAxSag).all():
    #             print('hey')
                
    #             finalListAx.append(pAxCor)
    #             pInCor  =  arrayCor[ipAxCor,:]
    #             pInSag = arraySag[ipAxSag,:]
    #             finalListCor.append(pInCor)
    #             finalListSag.append(pInSag)
    #             break
            
 

    return pointCoordinateInAx,pointCoordinateInCor,pointCoordinateInSag

def computeErrorOfRegistration(pointCoordinateInAx,pointCoordinateInCor,pointCoordinateInSag,SliceAx,SliceCor,SliceSag):
    
    nbpointError = pointCoordinateInAx.shape[0]  
    res = np.zeros(nbpointError)
    
    for ip in range(nbpointError):
            
          pAx = pointCoordinateInAx[ip,:]
          print('pAx : ', pAx)
        
          zax = int(pAx[2])
          Max = SliceAx[zax].get_transfo()
          p2Dax = np.zeros(4)
          p2Dax[0:2] = pAx[0:2]
          p2Dax[3] = 1 
          pInAx = Max @ p2Dax
        
          pCor = pointCoordinateInCor[ip,:]   
          print('pCor :', pCor)
        
          zcor = int(pCor[2])
          Mcor = SliceCor[zcor].get_transfo()
          p2Dcor = np.zeros(4)
          p2Dcor[0:2] = pCor[0:2]
          p2Dcor[3] = 1
          pInCor = Mcor @ p2Dcor
        
          pSag = pointCoordinateInSag[ip,:] 
          print('pSag :', pSag)
                
          zsag = int(pSag[2])
          Msag = SliceSag[zsag].get_transfo()
          p2Dsag = np.zeros(4)
          p2Dsag[0:2] = pSag[0:2]
          p2Dsag[3] = 1
          pInSag = Msag @ p2Dsag
         
          print('Ax:', pInAx)
          print('Sag:', pInSag)
          print('Cor:', pInCor)
          
          diffAxSag = np.sqrt((pInAx[0]-pInSag[0])**2 + (pInAx[1]-pInSag[1])**2 + (pInAx[2]-pInSag[2])**2)
          diffAxCor = np.sqrt((pInAx[0]-pInCor[0])**2 + (pInAx[1]-pInCor[1])**2 + (pInAx[2]-pInCor[2])**2)
          diffSagCor = np.sqrt((pInSag[0]-pInCor[0])**2 + (pInSag[1]-pInCor[1])**2 + (pInSag[2]-pInCor[2])**2)
          perimetre  = (diffAxSag + diffAxCor + diffSagCor)/3
          res[ip] = perimetre
    
    return res

def ErrorInParametersEstimation(motion_parameters,new_parameters):
    """
    
    The function computes the error between the parameters estimated in the reconstruction and the true parameters
    
    Inputs : 
    motion_parameters : Simulated parameters of motion
    new_parameters : Parameters obtained with the reconstruction algorithm
    
    Outputs : 
    EvolutionErrorParameters : Arrray of difference in parameters estimation

    """
    nit, nbparameters,nbslice = new_parameters.shape
    EvolutionErrorParameters=np.zeros((nit,nbparameters,nbslice))

    
    for i in range(nit): #for each iteration of the reconstruction algorithm
        EvolutionErrorParameters[i,:,:] = abs(motion_parameters[:,:] +  new_parameters[i,:,:]) #We compute the difference between the parameters estimated and the true parameters
        #The parameters estimated must be the opposite of the true parameters
    return EvolutionErrorParameters

def SimulImageWth0Transform(listSlice):
    """
    The function creates new slices from the one with movements, the new slice has an identity rigid transformation. This allow a more realistic representation of the data

    Inputs :
    listSlice : a list of sliceObject containing the slices with simulated movement. The rigid transformation corresponds to the simulated translation and rotation

    Outputs : 
    newlistSlice : a list of sliceObject containing the slices with simulated movement. The affine transformation of each slices is modified so that the rigid transform of the sliceObject is the identity
    """
    newlistSlice = []
    for s in listSlice: #for each elements in the list
        imgNifti = nib.Nifti1Image(s.get_slice().get_fdata(), s.get_transfo()) #create a new Nifti_Image, this create an image from the slices with movement with a new affin transform.
        #The new affine is composed of the previous affine but also of the translation and rotation.
        mask = nib.Nifti1Image(s.get_mask(),s.get_transfo())
        loadSlice(imgNifti,mask,newlistSlice,s.get_orientation()) #create a new sliceObject from imgNifti
    return newlistSlice
        
def createSlices(img,mask):
    """
    The function creates a volume with his coordinates in X,Y,Z where Z is the smaller resolution. Z represents the number of slices

    Inputs :
    img : Image in 3D
    mask : Mask of the brain associated to the image
   
    Returns :
    img : Image in X,Y,Z where Z is the smaller dimension
    msk : Mask of the brain associated with the image
    """
    sliceRes = 1 #number of slice to subsample the image, in the final volume, we take only 1 over sliceRes images
    imgShape = img.shape
    minShape = min(imgShape)
    
    if minShape == imgShape[0]: #if the smaller resolution is the first resolution (sagittal)
        X = imgShape[2]
        Y = imgShape[1]
        Z = int(np.floor(imgShape[0])) 
        newImage = np.zeros((X,Y,Z))
        newMask = np.zeros((X,Y,Z))
        currentAffine = img.affine
        Transfo = np.array([[0,0,sliceRes,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]) #Rotation matrix to keep the image in the same orientation
        newAffine = currentAffine @ Transfo
        #print(newAffine)
        for z in range(minShape):
            newImage[:,:,z] = np.transpose(img.get_fdata()[z,:,:])
            newMask[:,:,z] = np.transpose(mask.get_fdata()[z,:,:])
        
        img = nib.Nifti1Image(newImage,newAffine)
        msk = nib.Nifti1Image(newMask,newAffine)
        
    elif minShape == imgShape[1]: #if the smaller resolution is the second resolution (coronal)
        X = imgShape[0]
        Y = imgShape[2]
        Z = int(np.floor(imgShape[1]/sliceRes))
        newImage = np.zeros((X,Y,Z))
        newMask = np.zeros((X,Y,Z))
        currentAffine = img.affine
        Transfo = np.array([[1,0,0,0],[0,0,sliceRes,0],[0,1,0,0],[0,0,0,1]]) #Rotation matrix to keep the image in the same orientation
        newAffine = currentAffine @ Transfo 
        for z in range(minShape):
            newImage[:,:,z] = img.get_fdata()[:,z,:]
            newMask[:,:,z] = mask.get_fdata()[:,z,:]
        
        img = nib.Nifti1Image(newImage,newAffine)
        msk = nib.Nifti1Image(newMask,newAffine)
            
    else : #if the smaller resolution is the third resolution (Axial)
        X = imgShape[0]
        Y = imgShape[1]
        Z = int(np.floor(imgShape[2]/sliceRes)) #Rotation matrix to keep the image in the same orientation
        newImage = np.zeros((X,Y,Z))
        newMask = np.zeros((X,Y,Z))
        currentAffine = img.affine
        Transfo = np.array([[1,0,0,0],[0,1,0,0],[0,0,sliceRes,0],[0,0,0,1]])
        newAffine = currentAffine @ Transfo
        
        for z in range(minShape):
            newImage[:,:,z] = img.get_fdata()[:,:,z]
            newMask[:,:,z] = mask.get_fdata()[:,:,z]
        
        img = nib.Nifti1Image(newImage,newAffine)
        msk = nib.Nifti1Image(newMask,newAffine)
    
    return img,msk
            


def createMvt(listSlice,bounds):
    """
    The function create mouvement bteween the slices of a 3D mri image

    Inputs :
    listSlice : listSlice containing the original slices, with no movement

    Returns
    motion_parameters : the random motion parameters for each slices

    """
    nbSlice = len(listSlice)
    boundsAngle = bounds #in degrees
    boundsTranslation = bounds #in mm
    rangeAngle = boundsAngle[1]-boundsAngle[0]
    rangeTranslation = boundsTranslation[1]-boundsAngle[0]
    motion_parameters = np.zeros((6,nbSlice))
    
    i = 0
    for s in listSlice:   
        x = 0
        a1 = rd.random()*(rangeAngle) - (rangeAngle)/2
        a2 = rd.random()*(rangeAngle) - (rangeAngle)/2
        a3 = rd.random()*(rangeAngle) - (rangeAngle)/2
        t1 = rd.random()*(rangeTranslation) - (rangeTranslation)/2
        t2 = rd.random()*(rangeTranslation) - (rangeTranslation)/2
        t3 = rd.random()*(rangeTranslation) - (rangeTranslation)/2
        x = [a1,a2,a3,t1,t2,t3]
        s.set_parameters(x)
        motion_parameters[:,i]=x
        i=i+1
    return motion_parameters


def create3VolumeFromAnImage(image):
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
    sliceRes = 5 #number of slice to subsample the image, in the final volume, we take only 1 over sliceRes images
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
    for index in vectz: #Create the axial image
        img_axial[:,:,i] = image.get_fdata()[:,:,index]
        i=i+1
    imageAffine = image.affine
    
    
    transfoAx = np.array([[1,0,0,0],[0,1,0,0],[0,0,sliceRes,0],[0,0,0,1]]) #Rotation to obtain an axial orientation
    axAffine = imageAffine @ transfoAx
    
    i=0
    for index in vecty:
        img_coronal[:,:,i] = image.get_fdata()[:,index,:]
        i=i+1

    transfoCor = np.array([[1,0,0,0],[0,0,sliceRes,0],[0,1,0,0],[0,0,0,1]]) #Rotation to obtain a coronal orientation
    corAffine =  imageAffine @ transfoCor

    
    i=0
    for index in vectx:
        img_sagittal[:,:,i] = image.get_fdata()[index,:,:].T
        i=i+1

    transfoSag = np.array([[0,0,sliceRes,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]) #Rotation to obtain a sagittal orientation
    sagAffine =   imageAffine @ transfoSag

    Volumeaxial = nib.Nifti1Image(img_axial,  axAffine)
    VolumeCoronal = nib.Nifti1Image(img_coronal, corAffine)
    VolumeSagittal = nib.Nifti1Image(img_sagittal, sagAffine)
    
    return Volumeaxial,VolumeCoronal,VolumeSagittal      

