# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Jun 21 13:42:38 2022

# @author: mercier
# """
import SimpleITK as sitk
import numpy as np
import os
from os.path import exists
import joblib
from tools import createVolumesFromAlist, rigidMatrix, ParametersFromRigidMatrix
from data_simulation import ErrorOfRegistrationBtw2Slice, findCommonPointbtw2V, ChamferDistance, createArrayOfChamferDistance, ErrorOfRegistrationBtw2SliceBySlice
import json
from load import loadSlice, loadimages
import nibabel as nib
import matplotlib.pyplot as plt
import registration
from outliers_detection_intersection import ErrorSlice, createVolumesFromAlistError

def convert2EbnerParam(joblib,list_prefix,directory):
    """
    Save rigid transformations for each slices, execept rejected slices, into a directory. The directory can be used in Ebner reconstruction.

    Inputs :

    joblib : result of the registration
    list_prefix : prefix of images
    directory :  saving directory
    
    """
    
    key=[p[0] for p in joblib]
    element=[p[1] for p in joblib]

    
    listSlice=element[key.index('listSlice')]  #list of slice
    parameters=element[key.index('EvolutionParameters')][-1,:,:] #estimated paramters of the registration
    rejectedSlice=element[key.index('RejectedSlices')] #rejected slices

    
    images,mask = createVolumesFromAlist(listSlice.copy()) #list of images corresponding to differents original stacks
    
    for i_slice in range(len(listSlice)): 
        slicei=listSlice[i_slice]
        slicei.set_parameters(parameters[i_slice,:]) #set parameters to the last estimated parameters
        
    mat = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]) #matrix to convert affine matrix from nibabel to itk

    for n in range(len(images)): #for each stack
        
        imagen = images[n]
       
        for i_slice in range(len(images[n])): #for each slices (in each stacks)
            
            slicei=imagen[i_slice]
            s = (slicei.get_orientation(),slicei.get_index_slice())
            
            if s not in rejectedSlice:
                dimension=3
                X,Y,Z= slicei.get_slice().get_fdata().shape
                center= -slicei.get_center()
                centerMat = np.eye(4)
                centerMat[0:3,3] = center[0:3]
                invcenterMat = np.eye(4)
                invcenterMat[0:3,3] = -center[0:3]    
                p = slicei.get_parameters()
                matrix = mat @  centerMat @ rigidMatrix([p[0],p[1],p[2],p[3],p[4],p[5]]) @ invcenterMat  @ mat
                test = sitk.AffineTransform(dimension)
                test.SetMatrix(matrix[0:3,0:3].flatten())
                test.SetTranslation(matrix[0:3,3])
                images_index = slicei.get_index_image()
                sitk.WriteTransform(test,"%s/%s_slice%d.tfm" %(directory,list_prefix[images_index],slicei.get_index_slice())) #save rigid transformation, computed at the barycenter of the image, adatpted to itk
            #else:
                #print(s)

def convert2EbnerParamOriginalParam(listSlice,list_prefix,directory,paramAx,paramCor,paramSag):
    """
    Save rigid transformations for each slices, execept rejected slices, into a directory. The directory can be used in Ebner reconstruction.

    Inputs :

    joblib : result of the registration
    list_prefix : prefix of images
    directory :  saving directory
    
    """
    paramAx=np.load(paramAx)
    paramCor=np.load(paramCor)
    paramSag=np.load(paramSag)
    param=[]
    param.append(paramAx)
    param.append(paramCor)
    param.append(paramSag)
    
    images,mask = createVolumesFromAlist(listSlice.copy()) #list of images corresponding to differents original stacks
    
        
    mat = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]) #matrix to convert affine matrix from nibabel to itk

    for n in range(len(images)): #for each stack
        
        imagen = images[n]
       
        for i_slice in range(len(images[n])): #for each slices (in each stacks)
            
            slicei=imagen[i_slice]
            dimension=3
            X,Y,Z= slicei.get_slice().get_fdata().shape
            transfo = param[n][slicei.get_index_slice(),:,:]
            #print()
            matrix = mat @  transfo  @ mat
            #print(matrix)
            test = sitk.AffineTransform(dimension)
            test.SetMatrix(matrix[0:3,0:3].flatten())
            test.SetTranslation(matrix[0:3,3])
            images_index = slicei.get_index_image()

            sitk.WriteTransform(test,"%s/%s_slice%d.tfm" %(directory,list_prefix[images_index],slicei.get_index_slice())) #save rigid transformation, computed at the barycenter of the image, adatpted to itk


def computeRegErrorEbner(dir_motion,listSlice,listSlice_nomvt,transfo,prefix):
    
    images,mask=createVolumesFromAlist(listSlice_nomvt.copy())
    
    file_rejectedSlices = dir_motion + '/rejected_slices.json'
    
    rejectedSlices=[]

    if  exists(file_rejectedSlices):
        with open(file_rejectedSlices) as rejectedjson:
            data = json.load(rejectedjson)
            
        i=0
        key=data.keys()
        for k in key:
            for ek in data[k]:
                e=(i,ek)
                rejectedSlices.append(e)
            i=i+1
        #print(rejectedSlices)
    
    listptimg1img2_img1=[];listptimg1img2_img2=[]
    for i1 in range(len(images)):
        for i2 in range(len(images)):
            if i1 < i2:
               ptimg1img2_img1, ptimg1img2_img2 = findCommonPointbtw2V(images[i1],images[i2],rejectedSlices) #common points between volumes when no movement
               listptimg1img2_img1.append(ptimg1img2_img1)
               listptimg1img2_img2.append(ptimg1img2_img2)
    

    director = os.listdir(dir_motion); files = []
    nbSlice=len(listSlice)
    mat = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    for file in director:
        files.append(file)

    for i_slice in range(nbSlice):
        s=listSlice[i_slice]

        volume_index = s.get_orientation()
        for file in files:   

            if prefix[volume_index] in file :
                
                name_volume = prefix[volume_index]
                num_slice = file.replace(name_volume,'')
                num_slice = num_slice.replace('_slice','')
                num_slice = num_slice.replace('.tfm','')
 
                matrix_volume = sitk.ReadTransform(dir_motion + '/' + name_volume + '.tfm')
                matrix_volume = matrix_volume.Downcast()
                rotation_volume  = matrix_volume.GetMatrix()
                numpy_rotation_volume=np.array(rotation_volume)
                numpy_rotation_volume=numpy_rotation_volume.reshape(3,3)
                translation_volume=matrix_volume.GetTranslation()
                numpy_translation_volume=np.array(translation_volume)
                volume=np.eye(4,4)
                volume[0:3,0:3]=numpy_rotation_volume
                volume[0:3,3]=numpy_translation_volume
                
                if num_slice != '':
                    if s.get_index_slice() == int(num_slice) :
                        #print('num :',num_slice)
                        #print('index :',i_slice)
                        matrix=sitk.ReadTransform(dir_motion + '/' + file)
                        matrix = matrix.Downcast()
                        Rot=matrix.GetMatrix()
                        npRot=np.array(Rot)
                        npRot=npRot.reshape(3,3)
                        Trans=matrix.GetTranslation()
                        npTrans=np.array(Trans)
                        M=np.eye(4,4)
                        M[0:3,0:3]=npRot
                        M[0:3,3]=npTrans
                        center= -s.get_center() 
                        centerMat = np.eye(4)
                        centerMat[0:3,3] = center[0:3] 
                        invcenterMat = np.eye(4)
                        invcenterMat[0:3,3] = -center[0:3]  
                        voxelTranslation = np.eye(4)
                        voxelTranslation[0:3,3] = -0.25
                        Minter = np.linalg.inv(invcenterMat) @  np.linalg.inv(mat) @ np.linalg.inv(volume) @ M  @ np.linalg.inv(mat) @ np.linalg.inv(centerMat) 
                        #np.linalg.inv(invcenterMat) @ np.linalg.inv(mat) @ np.linalg.inv(volume) @ M  @ np.linalg.inv(mat) @ np.linalg.inv(centerMat)
                        x=ParametersFromRigidMatrix(Minter)
                        #print('volume', name_volume, 'num', num_slice, 'x :', x)
                        #x = np.array([0,0,0,0,0,0])
                        s.set_parameters(x)
                        
                        #print(s.get_transfo())
                        #listCorrected.append(s) #list with the corrected parameters
    

    images_corrected, masks_corrected = createVolumesFromAlist(listSlice)
    
    listError=[]
    for i_slice in range(len(listSlice)):
        slicei=listSlice[i_slice]
        orientation = slicei.get_orientation()
        index = slicei.get_index_slice()
        Errori = ErrorSlice(orientation,index)
        listError.append(Errori)
    
    images_corrected, masks_corrected = createVolumesFromAlist(listSlice)
    listErrorVolume = createVolumesFromAlistError(listError)
    
    indice=0;listErrorAfter=[]
    for i1 in range(len(images)):
        for i2 in range(len(images)):
            if i1 < i2:
                transfo1 = np.load(transfo[i1])
                #print(transfo1)
                transfo2 = np.load(transfo[i2])
                #error of registration between volumes after registration
                errorimg1img2_after = ErrorOfRegistrationBtw2SliceBySlice(listptimg1img2_img1[indice],listptimg1img2_img2[indice],images_corrected[i1],images_corrected[i2],np.linalg.inv(transfo1),np.linalg.inv(transfo2),listErrorVolume[i1],listErrorVolume[i2])
                indice=indice+1
    
    listErrorAfter=[]
    for n_image in range(len(listErrorVolume)):
         listVolume = listErrorVolume[n_image]
         listv12 = []
         for i_slice in range(len(listVolume)):
             vol = listVolume[i_slice]
             listv12.append(vol.get_error())
         listErrorAfter.append(listv12)          

    
    return listErrorAfter#,listSlice

def displayErrorOfRegistration(listSlice,listErrorBefore,listErrorAfter,listColorMap): 
           
    images,masks=createVolumesFromAlist(listSlice)
    
    div=50;i=0
    for i1 in range(len(images)):
        for i2 in range(len(images)):
            if i1<i2:
                ErrorBefore = listErrorBefore[i][0::div]
                ErrorAfter = listErrorAfter[i][0::div]
                colormap = listColorMap[i][0::div]
                i=i+1   
                fig, axe = plt.subplots()
                im = axe.scatter(ErrorAfter,ErrorBefore,marker='.',c=colormap)
                cbar = fig.colorbar(im,ax=axe)
                plt.ylabel('before reg')
                plt.xlabel('after reg')
                title='%d and %d' %(i1,i2)
                plt.title(title) 
                plt.show()

                plt.figure()
                plt.subplot(121)
                plt.hist(ErrorBefore,range=(min(ErrorBefore),max(ErrorBefore)),bins='auto')
                title='%d and %d, \n before registration' %(i1,i2)
                plt.title(title)
                plt.subplot(122)
                plt.hist(ErrorAfter,range=(min(ErrorAfter),max(ErrorAfter)),bins='auto')
                plt.show()

                mean_before_ac = np.mean(ErrorBefore)
                std_before_ac = np.std(ErrorBefore)
                mean_after_ac = np.mean(ErrorAfter)
                std_after_ac = np.std(ErrorAfter)

                strbr = 'before registration : %f +/- %f'  %(mean_before_ac,std_before_ac)
                print(strbr)
                strar = 'after registration : %f +/- %f' %(mean_after_ac,std_after_ac)
                print(strar)

