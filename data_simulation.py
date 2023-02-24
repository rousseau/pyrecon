#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:38:29 2022

@author: mercier

This script aims to create motion simulation on an MRI image to validate the registration algorithm

"""

import numpy as np
import random as rd
from registration import commonSegment
from scipy.ndimage import map_coordinates
from scipy.ndimage import distance_transform_cdt
import tools 
import outliers_detection_intersection



    
def findCommonPointbtw2V(Volume1,Volume2,rejectedSlices):
    
    listPointVolume1 = []
    listPointVolume2 = []
    
    for zV1 in range(len(Volume1)):
        sV1 = Volume1[zV1]
        maskV1 = sV1.get_mask()
        for zV2 in range(len(Volume2)):
            sV2 = Volume2[zV2]
            maskV2 = sV2.get_mask()
            
            #print((sV1.get_orientation(),sV1.get_index_slice()))
            if ((sV1.get_orientation(),sV1.get_index_slice()) not in rejectedSlices) and ((sV2.get_orientation(),sV2.get_index_slice()) not in rejectedSlices) :
                #print('please work')
                #print(zV1)
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
                
                #print(np.shape((np.where(interpolMaskV1>0))),np.shape((np.where(interpolMaskV2>0))))
                if interpolMaskV1.any() or interpolMaskV2.any() : 
                    #print(np.where(interpolMaskV1)[0],np.where(interpolMaskV2)[0])
                    pV1 = pointV1[np.concatenate((np.where(interpolMaskV1)[0],np.where(interpolMaskV2)[0]))]
                    listpoint = pV1.tolist()
                    listPointVolume1.extend(listpoint)
                    
                    pV2 = pointV2[np.concatenate((np.where(interpolMaskV1)[0],np.where(interpolMaskV2)[0]))]
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
    
    inv_chamfer_distance = distance_transform_cdt(volume.get_fdata())

    return inv_chamfer_distance

def createArrayOfChamferDistance(ChamferDistance,coordinateInAx):
    
    dFromCenter = np.zeros(coordinateInAx.shape[0])
    indice = 0
    
    for c in coordinateInAx:    
        dFromCenter[indice] = ChamferDistance[int(c[0]),int(c[1]),int(c[2])]
        indice = indice + 1
    return dFromCenter

def createArrayOfRejectedSlices(mW,coordinateInAx):
    
    color = np.zeros(coordinateInAx.shape[0])
    indice=0
    for c in coordinateInAx:    
        color[indice] = mW
        indice = indice + 1
    return color
    
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

def ErrorOfRegistrationBtw2SliceBySlice(pointCoordinateV1,pointCoordinateV2,Volume1,Volume2,transfo1,transfo2,listErrorSliceV1,listErrorSliceV2):
    
    nbpointError = pointCoordinateV1.shape[0]
    res = np.zeros(nbpointError)
    
    for ip in range(nbpointError):
        
        pV1 = pointCoordinateV1[ip,:]
        #print('Hello:' ,pV1[0:2])
        pV2 = pointCoordinateV2[ip,:]
        
        zV1 = int(np.ceil(pV1[2]))
        zV2 = int(np.ceil(pV2[2]))
        

        Mv1 = transfo1[Volume1[zV1].get_index_slice(),:,:]  @ Volume1[zV1].get_transfo()
            #transfo1[Volume1[zV1].get_index_slice(),:,:]  @ Volume1[zV1].get_transfo()
        p2Dv1 = np.zeros(4)
        p2Dv1[0:2] = pV1[0:2]
        p2Dv1[3] = 1
        pInV1 = Mv1 @ p2Dv1
                
        
        Mv2 = transfo2[Volume2[zV2].get_index_slice(),:,:] @ Volume2[zV2].get_transfo()
        #transfo2[Volume2[zV2].get_index_slice(),:,:] @ Volume2[zV2].get_transfo()
        p2Dv2 = np.zeros(4)
        p2Dv2[0:2] = pV2[0:2]
        p2Dv2[3] = 1
        pInV2 = Mv2 @ p2Dv2
                
            #print(pInV1[2],pInV2[2])
        diff = np.sqrt((pInV1[0]-pInV2[0])**2 + (pInV1[1]-pInV2[1])**2 + (pInV1[2]-pInV2[2])**2)
                
        #zV1 est l indice de la coupe dans l image d origine et pas l'indice dans la liste !!!!
            
        ErrorSliceV1 = listErrorSliceV1[zV1]
        ErrorSliceV1.add_registration_error(diff)
        
                
        ErrorSliceV2 = listErrorSliceV2[zV2]
        ErrorSliceV2.add_registration_error(diff)
                    
            
            
        res[ip]=diff
      
    return res


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


def compute_registration_error(listInput,listnomvt,rejected_slices,res_joblib,transfo):
    
        key=[p[0] for p in res_joblib]
        element=[p[1] for p in res_joblib]
        #listSlice=element[key.index('listSlice')]
        listSlice=[]
        for ii in range(len(listInput)):
            listSlice.append(listInput[ii].copy())
        parameters_slices = element[key.index('EvolutionParameters')][-1,:,:]
    
        images,mask = tools.createVolumesFromAlist(listnomvt.copy()) 
    
        listptimg1img2_img1=[];listptimg1img2_img2=[]
        for i1 in range(len(images)):
            for i2 in range(len(images)):
                if i1 < i2:
                   #rejected_slices=[]
                   print(rejected_slices)
                   ptimg1img2_img1, ptimg1img2_img2 = findCommonPointbtw2V(images[i1],images[i2],rejected_slices) #common points between volumes when no movement
                   listptimg1img2_img1.append(ptimg1img2_img1)
                   listptimg1img2_img2.append(ptimg1img2_img2)
        print('Shape :',np.shape(listptimg1img2_img1))           
        listError=[]
        for i_slice in range(len(listSlice)):
            slicei=listSlice[i_slice]
            #slicei.set_parameters(parameters_slices[i_slice])
            orientation = slicei.get_orientation()
            index = slicei.get_index_slice()
            Errori = outliers_detection_intersection.ErrorSlice(orientation,index)
            listError.append(Errori)
            
        images_corrected, masks_corrected = tools.createVolumesFromAlist(listSlice)
        listErrorVolume = outliers_detection_intersection.createVolumesFromAlistError(listError)
            
        listerrorimg1img2_after=[];
        indice=0
        for i1 in range(len(images)):
            for i2 in range(len(images)):
                if i1 < i2:
                   transfo1 = np.load(transfo[i1])
                   transfo2 = np.load(transfo[i2])
                   errorimg1img2_after = ErrorOfRegistrationBtw2SliceBySlice(listptimg1img2_img1[indice],listptimg1img2_img2[indice],images_corrected[i1],images_corrected[i2],np.linalg.inv(transfo1),np.linalg.inv(transfo2),listErrorVolume[i1],listErrorVolume[i2])
                   indice=indice+1
        
        for n_image in range(len(listErrorVolume)):
            listVolume = listErrorVolume[n_image]
            listv12 = []
            for i_slice in range(len(listVolume)):
                vol = listVolume[i_slice]
                listv12.append(vol.get_error())
            listerrorimg1img2_after.extend(listv12)
        
        return listerrorimg1img2_after

          
                
                
                

    
    