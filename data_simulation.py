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
from registration import loadSlice,commonSegment,sliceProfil,computeCostBetweenAll2Dimages,costFromMatrix
from scipy.ndimage import map_coordinates
from scipy.ndimage import distance_transform_cdt
import random as rd

def createVolumesFromAlist(listSlice):
    
    orientation = []; images=[]; mask=[]
    for s in listSlice:
        s_or = s.get_orientation()
        if s_or in orientation:
            index_orientation = orientation.index(s_or)
            images[index_orientation].append(s)
            mask[index_orientation].append(s.get_mask())
        else:
            orientation.append(s_or)
            images.append([])
            mask.append([])
            index_orientation = orientation.index(s_or)
            images[index_orientation].append(s)
            mask[index_orientation].append(s.get_mask())
                
    return images, mask

def findCommonPointbtw2V(Volume1,Volume2):
    
    listPointVolume1 = []
    listPointVolume2 = []
    
    for zV1 in range(len(Volume1)):
        sV1 = Volume1[zV1]
        maskV1 = sV1.get_mask()
        for zV2 in range(len(Volume2)):
            sV2 = Volume2[zV2]
            maskV2 = sV2.get_mask()
            
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
            interpolMask = np.zeros(nbpt)
            ptInterpol = np.zeros((3,nbpt))
            ptInterpol[0,:] = pointV1[:,0]
            ptInterpol[1,:] = pointV1[:,1]
            ptInterpol[2,:] = np.zeros(nbpt)
            map_coordinates(maskV1, ptInterpol, output=interpolMask, order=0, mode='constant',cval=np.nan,prefilter=False)
            pV1 = pointV1[np.where(interpolMask>0)]
            listpoint = pV1.tolist()
            listPointVolume1.extend(listpoint)
            
            
            pointV2 = np.zeros((nbpt,3))
            pointV2[:,0] = np.linspace(pt2[0,0],pt2[0,1],nbpt)
            pointV2[:,1] = np.linspace(pt2[1,0],pt2[1,1],nbpt)
            pointV2[:,2] = np.ones(nbpt)*zV2
            interpolMask = np.zeros(nbpt)
            ptInterpol = np.zeros((3,nbpt))
            ptInterpol[0,:] = pointV2[:,0]
            ptInterpol[1,:] = pointV2[:,1]
            ptInterpol[2,:] = np.zeros(nbpt)
            map_coordinates(maskV2, ptInterpol, output=interpolMask, order=0, mode='constant',cval=np.nan,prefilter=False)
            pV2 = pointV2[np.where(interpolMask>0)]
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
    
    # nbSlice = len(listSliceVol)
    # X,Y,Z = listSliceVol[0].get_slice().get_fdata().shape
    # imgChamferDistance = np.zeros((X,Y,nbSlice))
    
    # indice = 0
    # for s in listSliceVol:
    #     chamfer_distance =  distance_transform_cdt(s.get_slice().get_fdata())
    #     Vmax = np.max(chamfer_distance)
    #     inv_chamfer_distance = np.abs(chamfer_distance - Vmax)
    #     imgChamferDistance[:,:,indice] = inv_chamfer_distance
    #     indice = indice + 1
    
    inv_chamfer_distance = distance_transform_cdt(volume.get_fdata())

    return inv_chamfer_distance

def createArrayOfChamferDistance(ChamferDistance,coordinateInAx):
    
    dFromCenter = np.zeros(coordinateInAx.shape[0])
    indice = 0
    
    for c in coordinateInAx:    
        dFromCenter[indice] = ChamferDistance[int(c[0]),int(c[1]),int(c[2])]
        indice = indice + 1
    return dFromCenter

    
def ErrorOfRegistrationBtw2Slice(pointCoordinateV1,pointCoordinateV2,Volume1,Volume2):
    
    nbpointError = pointCoordinateV1.shape[0]
    res = np.zeros(nbpointError)
    
    for ip in range(nbpointError):
        
        pV1 = pointCoordinateV1[ip,:]
        
        zV1 = int(pV1[2])
        Mv1 = Volume1[zV1].get_transfo()
        p2Dv1 = np.zeros(4)
        p2Dv1[0:2] = pV1[0:2]
        p2Dv1[3] = 1
        pInV1 = Mv1 @ p2Dv1
        
        pV2 = pointCoordinateV2[ip,:]
        
        zV2 = int(pV2[2])
        Mv2 = Volume2[zV2].get_transfo()
        p2Dv2 = np.zeros(4)
        p2Dv2[0:2] = pV2[0:2]
        p2Dv2[3] = 1
        pInV2 = Mv2 @ p2Dv2
        
        #print(pInV1[2],pInV2[2])
        diff = np.sqrt((pInV1[0]-pInV2[0])**2 + (pInV1[1]-pInV2[1])**2 + (pInV1[2]-pInV2[2])**2)
        res[ip]=diff
      
    return res

def computeErrorOfRegistration(pointCoordinateInAx,pointCoordinateInCor,pointCoordinateInSag,SliceAx,SliceCor,SliceSag):
    
    nbpointError = pointCoordinateInAx.shape[0]  
    res = np.zeros(nbpointError)
    
    for ip in range(nbpointError):
            
          pAx = pointCoordinateInAx[ip,:]
          #print('pAx : ', pAx)
        
          zax = int(pAx[2])
          Max = SliceAx[zax].get_transfo()
          p2Dax = np.zeros(4)
          p2Dax[0:2] = pAx[0:2]
          p2Dax[3] = 1 
          pInAx = Max @ p2Dax
        
          pCor = pointCoordinateInCor[ip,:]   
          #print('pCor :', pCor)
        
          zcor = int(pCor[2])
          Mcor = SliceCor[zcor].get_transfo()
          p2Dcor = np.zeros(4)
          p2Dcor[0:2] = pCor[0:2]
          p2Dcor[3] = 1
          pInCor = Mcor @ p2Dcor
        
          pSag = pointCoordinateInSag[ip,:] 
          #print('pSag :', pSag)
                
          zsag = int(pSag[2])
          Msag = SliceSag[zsag].get_transfo()
          p2Dsag = np.zeros(4)
          p2Dsag[0:2] = pSag[0:2]
          p2Dsag[3] = 1
          pInSag = Msag @ p2Dsag
         
          #print('Ax:', pInAx)
          #print('Sag:', pInSag)
          #print('Cor:', pInCor)
          
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
            


def createMvt(listSlice,boundsRot,boundsTrans):
    """
    The function create mouvement bteween the slices of a 3D mri image

    Inputs :
    listSlice : listSlice containing the original slices, with no movement

    Returns
    motion_parameters : the random motion parameters for each slices

    """
    nbSlice = len(listSlice)
    boundsAngle = boundsRot #in degrees
    boundsTranslation = boundsTrans #in mm
    rangeAngle = boundsAngle[1]-boundsAngle[0]
    rangeTranslation = boundsTranslation[1]-boundsTranslation[0]
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
        x = np.array([a1,a2,a3,t1,t2,t3])
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
    sliceRes = 1 #number of slice to subsample the image, in the final volume, we take only 1 over sliceRes images
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



def createAnErrorImage(AxCoordinate,error,ImageSize):
    
    ImageError = np.zeros(ImageSize)
    indice = 0
    
    for p in AxCoordinate:
        #print('p : ',p.astype(int))
        #print('Error : ',error[indice])
        pt = p.astype(int)
        #print(pt[1])
        ImageError[pt[0],pt[1],pt[2]] = error[indice]
        indice=indice+1
    
    return ImageError
    
def displaySimulInIsoImage(listImgMvt,NoMvtAx,NoMvtCor,NoMvtSag,Iso):
    
    listImgNoMvtAx = []    
    loadSlice(NoMvtAx, None, listImgNoMvtAx, "Axial")
    listImgNoMvtCor = []
    loadSlice(NoMvtCor, None, listImgNoMvtCor, "Coronal")
    listImgNoMvtSag = []
    loadSlice(NoMvtSag, None, listImgNoMvtSag, "Sagittal")
    
    X,Y,Z = Iso.shape
    res = Iso.get_fdata()
    
    #dbug
    # gridError,gridNbpoint = computeCostBetweenAll2Dimages(listImgNoMvt,'MSE')
    # cost = costFromMatrix(gridError,gridNbpoint)
    
    # if cost > 0:
    #     print('cost :',cost)
    #     return 0
    
    for i_s1 in range(len(listImgMvt)):
        s1 = listImgMvt[i_s1]
       
        for i_s2 in range(len(listImgNoMvtAx)):
            s2 = listImgNoMvtAx[i_s2]
            
            sliceimage1=s1.get_slice().get_fdata();M1=s1.get_transfo();res=min(s1.get_slice().header.get_zooms())
            sliceimage2=s2.get_slice().get_fdata();M2=s2.get_transfo()
            ps1,ps2,nbpoint,ok = commonSegment(sliceimage1,M1,sliceimage2,M2,res)
            nbpoint=np.int32(nbpoint[0,0]);ok=np.int32(ok[0,0])
            
            if ok>0:
                profil,index,ndpoints1 = sliceProfil(s1,ps1,nbpoint)
                
                x = np.linspace(ps2[0,0],ps2[0,1],nbpoint,dtype=int)
                y = np.linspace(ps2[1,0],ps2[1,1],nbpoint,dtype=int)
                
                for index in range(len(x)):
                    
                    if((x[index]>=0) and (y[index]>=0) and (x[index]<s2.get_slice().get_fdata().shape[0]) and (y[index]<s2.get_slice().get_fdata().shape[1])):
    
                        res[x[index],y[index],i_s2] = profil[index]
              
        for i_s2 in range(len(listImgNoMvtCor)):
            s2 = listImgNoMvtCor[i_s2]
            ps1,ps2,nbpoint,ok = commonSegment(sliceimage1,M1,sliceimage2,M2,res)
            nbpoint=np.int32(nbpoint[0,0]);ok=np.int32(ok[0,0])
            
            if ok>0:
                profil,index,ndpoints1 = sliceProfil(s1,ps1,nbpoint)
                
                x = np.linspace(ps2[0,0],ps2[0,1],nbpoint,dtype=int)
                y = np.linspace(ps2[1,0],ps2[1,1],nbpoint,dtype=int)
                
                for index in range(len(x)):
                    
                    if((x[index]>=0) and (y[index]>=0) and (x[index]<s2.get_slice().get_fdata().shape[0]) and (y[index]<s2.get_slice().get_fdata().shape[1])):
                            
                            res[x[index],i_s2,y[index]] = profil[index]
                       
        for i_s2 in range(len(listImgNoMvtSag)):
            s2 = listImgNoMvtSag[i_s2]
            ps1,ps2,nbpoint,ok = commonSegment(sliceimage1,M1,sliceimage2,M2,res)
            nbpoint=np.int32(nbpoint[0,0]);ok=np.int32(ok[0,0])
            
            if ok>0:
                profil,index,ndpoints1 = sliceProfil(s1,ps1,nbpoint)
                
                x = np.linspace(ps2[0,0],ps2[0,1],nbpoint,dtype=int)
                y = np.linspace(ps2[1,0],ps2[1,1],nbpoint,dtype=int)
                
                for index in range(len(x)):
                    
                    if((x[index]>0) and (y[index]>0) and (x[index]<s2.get_slice().get_fdata().shape[0]) and (y[index]<s2.get_slice().get_fdata().shape[1])):
                               
                            res[i_s2,y[index],x[index]] = profil[index]
    
    resNifti = nib.Nifti1Image(res, Iso.affine)           
    return resNifti            
                
                
                

    
    