# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Jun 21 13:42:38 2022

# @author: mercier
# This script converts the results stored in ROSI into parameters that can be used with NiftyMIC.

# """
import SimpleITK as sitk
import numpy as np
import os
from os.path import exists
from rosi.registration.tools import separate_slices_in_stacks
from rosi.registration.transformation import rigidMatrix, ParametersFromRigidMatrix
from rosi.simulation.validation import tre_indexes, slice_tre
import json
import matplotlib.pyplot as plt
from rosi.registration.outliers_detection.outliers import sliceFeature, separate_features_in_stacks
from rosi.registration.tools import same_order
import nibabel as nib
from rosi.registration.sliceObject import SliceObject
from math import isclose
import re

def convert2EbnerParam(joblib,list_prefix,directory):
    """
    Save rigid transformations for each slice, except rejected slices, in a directory. The directory can be used in Ebner (NiftyMIC) reconstruction.

    Inputs :

    joblib : result of ROSI (ex : res.joblib.gz)
    list_prefix : name of the LR images (ex : name for the image name.nii.gz)
    /!\ elements in list_prefix must be exactly the same than the LR images name 
    directory : output directory (ex : motion_correction)
    
    """
    
    key=[p[0] for p in joblib]
    element=[p[1] for p in joblib]

    
    listSlice=element[key.index('listSlice')]  #list of slice
    parameters=element[key.index('EvolutionParameters')][-1,:,:] #estimated parameters of the registration
    rejectedSlice=element[key.index('RejectedSlices')] #rejected slices
    #print(rejectedSlice)

    
    images,mask = separate_slices_in_stacks(listSlice.copy()) #create one list per LR image
    
    for i_slice in range(len(listSlice)): 
        slicei=listSlice[i_slice]
        slicei.set_parameters(parameters[i_slice,:]) #set parameters to the last estimated parameters in ROSI
        
    mat = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]) #matrix to convert affine matrix from nibabel to itk

    for n in range(len(images)): #for each stack
        
        imagen = images[n]
        

       
        for i_slice in range(len(images[n])): #for each slices (in each stacks)
            
            slicei=imagen[i_slice]    
            s = (slicei.get_indexVolume(),slicei.get_indexSlice())
            #print(s)
            #print(s in rejectedSlice)
            
            if s not in rejectedSlice:
                dimension=3
                X,Y,Z= slicei.get_slice().get_fdata().shape
                center= -slicei.get_centerOfRotation()
                centerMat = np.eye(4)
                centerMat[0:3,3] = center[0:3]
                invcenterMat = np.eye(4)
                invcenterMat[0:3,3] = -center[0:3]    
                p = slicei.get_parameters()
                matrix = mat @  centerMat @ rigidMatrix([p[0],p[1],p[2],p[3],p[4],p[5]]) @ invcenterMat  @ mat  #conversion from ROSI to NiftyMIC
                test = sitk.AffineTransform(dimension)
                test.SetMatrix(matrix[0:3,0:3].flatten())
                test.SetTranslation(matrix[0:3,3])
                images_index = slicei.get_indexVolume()
                sitk.WriteTransform(test,"%s/%s_slice%d.tfm" %(directory,list_prefix[images_index],slicei.get_indexSlice())) #write converted transformation into the output directory
            #else:
                #print(s)


def computeRegErrorEbner(dir_motion,listSlice,listSlice_nomvt,transfo,prefix):
    """
    This function compute the Median Registration Error (RTRE) using output from NiftyMIC

    Inputs :
    dir_motion : directory of NiftyMIC output transformations
    listSlice : list of slices from simulated LR images
    listSlice_nomvt : list of slices from the LR images simulated with no motion
    transfo : simulated transformation, stored in a numpy array (e.g. : LrAx_transfo.npy)
    prefix : name of the LR images (ex : name for image name.nii.gz)

    Outputs : 
    listErrorAfter : list of the RTRE for each slice
    listSlice : list of slices from simulated LR image
    """

    images,mask=separate_slices_in_stacks(listSlice_nomvt.copy())
    
    file_rejectedSlices = dir_motion + '/rejected_slices.json' #in niftyMIC, index of rejected slices are stored in a json file
    
    rejectedSlices=[] 

    
    director = os.listdir(dir_motion); files = []
    print(director)
    nbSlice=len(listSlice)
    mat = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    for file in director:
        files.append(file)

    for i_slice in range(nbSlice):
        s=listSlice[i_slice]
        volume_index = s.get_stackIndex()     
        name = prefix[volume_index] + '_slice' + str(s.get_indexSlice()) + '.tfm'
        
        
        if name not in files: #If a slice does not have a corresponding transformation in the directory, it is rejected.
            rejectedSlices.append((volume_index,s.get_indexSlice()))
        else :   
        
                
            name_volume = prefix[volume_index]
              
            if exists(dir_motion + '/' + name_volume + '.tfm'): 
                #In NiftyMic, the first step is a VVR (Volume to Volume Registration).
               
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
            
            else :
                volume=np.eye(4,4) #by default, the transformation of the volume is the identiy matrix
                
            matrix=sitk.ReadTransform(dir_motion + '/' + name)
            matrix = matrix.Downcast()
            Rot=matrix.GetMatrix()
            npRot=np.array(Rot)
            npRot=npRot.reshape(3,3)
            Trans=matrix.GetTranslation()
            npTrans=np.array(Trans)
            M=np.eye(4,4)
            M[0:3,0:3]=npRot
            M[0:3,3]=npTrans
            center= -s.get_centerOfRotation() 
            centerMat = np.eye(4)
            centerMat[0:3,3] = center[0:3] 
            invcenterMat = np.eye(4)
            invcenterMat[0:3,3] = -center[0:3]  
            Minter = np.linalg.inv(centerMat) @  np.linalg.inv(mat) @ np.linalg.inv(volume) @ M  @ np.linalg.inv(mat) @ np.linalg.inv(invcenterMat) #Parameters are converted to the ROSI system to facilitate TRE calculation.
               
            x=ParametersFromRigidMatrix(Minter)

            s.set_parameters(x)
            print(s.get_parameters())
            print(listSlice[i_slice].get_parameters())
                        

    #TRE computation
    #probablement possible de factoriser le calcul ...
    listptimg1img2_img1=[];listptimg1img2_img2=[]
    for i1 in range(len(images)):
        for i2 in range(len(images)):
            if i1 < i2:
               transfo1 = np.load(transfo[i1])
               transfo2 = np.load(transfo[i2])
               print('sV1 ',images[i1][-1])
               ptimg1img2_img1, ptimg1img2_img2 = tre_indexes(images[i1],images[i2],transfo1,transfo2,rejectedSlices) #common points between volumes when no movement
               listptimg1img2_img1.append(ptimg1img2_img1)
               listptimg1img2_img2.append(ptimg1img2_img2)
    

    images_corrected, masks_corrected = separate_slices_in_stacks(listSlice)
    
    listError=[]
    for i_slice in range(len(listSlice)):
        slicei=listSlice[i_slice]
        #print(slicei.get_parameters())
        orientation = slicei.get_stackIndex()
        index = slicei.get_indexSlice()
        Errori = sliceFeature(orientation,index)
        listError.append(Errori)
    
    images_corrected, masks_corrected = separate_slices_in_stacks(listSlice)
    listErrorVolume = separate_features_in_stacks(listError) #list to savec error for each slice   
    indice=0;listErrorAfter=[]
    for i1 in range(len(images)):
        for i2 in range(len(images)):
            if i1 < i2:
                transfo1 = np.load(transfo[i1])
                transfo2 = np.load(transfo[i2])
                
                #error of registration between volumes after registration
                errorimg1img2_after = slice_tre(listptimg1img2_img1[indice],listptimg1img2_img2[indice],images_corrected[i1],images_corrected[i2],listErrorVolume[i1],listErrorVolume[i2])
                indice=indice+1
    
    listErrorAfter=[]
    for n_image in range(len(listErrorVolume)):
         listVolume = listErrorVolume[n_image]
         listv12 = []
         for i_slice in range(len(listVolume)):
             vol = listVolume[i_slice]
             listv12.append(vol.get_median_error())
         listErrorAfter.append(listv12)          

    
    return listErrorAfter,listSlice,rejectedSlices

def displayErrorOfRegistration(listSlice,listErrorBefore,listErrorAfter,listColorMap): 
    """
    Function that display the TRE in a beautiful graphic :) 
    (presents the TRE after registration in function of the TRE before registration)
    
    Inputs:
    listSlice : list of slices from LR images
    listErrorBefore : TRE for each slice before registration
    listErrorAfter : TRE for each slice after registration
    listColorMap : colouring of the graphic 
    (for a nicer graphic)
    """           


    images,masks=separate_slices_in_stacks(listSlice)
    
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



def which_stack(affine_matrix,slice_thickness):
    """
    Function to determine the stack corresponding to a slice, based on its affine matrix and slice thickness.
    """
    
    resolution = affine_matrix[:,2]
    
    if isclose(np.abs(resolution[2]),slice_thickness,abs_tol=1) :  
        return 0
    elif isclose(np.abs(resolution[1]),slice_thickness,abs_tol=1):
        return 1
    elif isclose(np.abs(resolution[0]),slice_thickness,abs_tol=1)  :
        return 2
    else :
        print("The function wasn't able to find the stack and return -1")
        print(resolution)
        return -1

def where_in_the_stack(stack_affine,slice_affine,num_stack):
    """
    Function to determine the slice index within a stack.
    """

    inv_matrice = np.linalg.inv(stack_affine) @ slice_affine
    index_in_stack = np.int32(np.round(inv_matrice[2,3]))


    print('index :',index_in_stack)
    return index_in_stack


def sorted_alphanumeric(data):
    """
    Function to sort files in a directory alphanumerically.
    """
    
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def convert2ListSlice(dir_nomvt,dir_slice,slice_thickness,set_of_affines):
    """
    Function to convert corrected slices output by svort or nesvor into sliceObjects.

    Inputs:
    dir_slice: Path to the directory containing motion-corrected slices (output of Svort/Nesvor).
    dir_nomvt: Path to the directory containing slices without simulated motion.
    set_of_affines: List of affine matrices representing the original low-resolution (LR) image transformations.
    slice_thickness: Thickness of the slices


    Outputs:
    listnomvt: List of SliceObject instances generated from the dir_nomvt directory.
    listSlice:  List of SliceObject instances generated from the dir_slice directory.

    N.B : The dir_nomvt directory is required to determine the slice index and stack index for each SliceObject during conversion.
    """


    list_file = sorted_alphanumeric(os.listdir(dir_slice))


    listSlice=[];listnomvt=[]
    index=np.zeros(3,np.int32)
    for file in list_file:
        #check that the file is a slice
        
        if not 'mask' in file and not 'image' in file and not 'volume' in file : 
            ##else do nothing
            print('file_name :',file)

            slice_data=nib.load(dir_slice + '/' + file)
            mask_data=nib.load(dir_slice + '/mask_' + file)
            slice_nomvt = nib.load(dir_nomvt + '/' + file)
            mask_slice=nib.load(dir_slice + '/mask_' + file) 

            num_stack=which_stack(slice_nomvt.affine,slice_thickness) 
            stack_affine=set_of_affines[num_stack] #determine the stack of the slice
            i_slice=where_in_the_stack(stack_affine,slice_nomvt.affine,num_stack) #determine the slice index

            slicei = SliceObject(slice_data,mask_data.get_fdata(),num_stack,i_slice, num_stack)
            slicen = SliceObject(slice_nomvt,mask_slice.get_fdata(),num_stack,i_slice, num_stack)
                
            listSlice.append(slicei)
            listnomvt.append(slicen)
    
    return listnomvt,listSlice
    

def remove_slices_with_small_mask(listSlice):
    """
    Function to remove slices with excessively small masks, as they cannot be registered.
    """
    
    print(type(listSlice[0]))
    images,_=separate_slices_in_stacks(listSlice)
    listres=[]
    for stack in images:
        max_mprop = np.max([np.sum(islice.get_mask())/islice.get_slice().get_fdata().size for islice in stack])
        [stack.remove(islice) for islice in stack if (np.sum(islice.get_mask())/islice.get_slice().get_fdata().size)/max_mprop<0.1]
        listres.extend(stack)
    return listres


def computeRegErrorNesVor(dir_slice,slice_thickness,dir_nomvt,set_of_affines,transfo):
    """
    Function to compute the Robust Target Registration Error (RTRE) or median TRE on simulated motion-corrected data (Nesvor or Svort).

    Inputs:
    dir_slice: Path to the directory containing motion-corrected slices (output of Svort/Nesvor).
    slice_thickness: Thickness of the slices
    dir_nomvt: Path to the directory containing slices without simulated motion.
    set_of_affines: List of affine matrices representing the original low-resolution (LR) image transformations.
    transfo: Theoretical transformation matrix.


    Outputs:
    Median Target Registration Error (or RTRE) computed from the corrected data.

    N.B : The dir_nomvt directory and transfo are required to determine the theoretical intersection points.
    """
    
    listSlice_nomvt,listSlice=convert2ListSlice(dir_nomvt,dir_slice,slice_thickness,set_of_affines) #convert the results from Svort/Nesvor into a list of sliceObject
    print(listSlice)
    listSlice = remove_slices_with_small_mask(listSlice) #slices with excessively small masks
    listSlice_nomvt = remove_slices_with_small_mask(listSlice_nomvt) 
       
    images,_=separate_slices_in_stacks(listSlice) #Verify the correspondence between slices with corrected motion and slices without motion.
    images_order = [images[i][0].get_stackIndex() for i in range(0,len(images))]
    index_im=np.argsort(images_order)
    print(index_im)
    nomvt,_= separate_slices_in_stacks(listSlice_nomvt)
    images = [images[x] for x in index_im]
    nomvt = [nomvt[x] for x in index_im]   
    
    
    listptimg1img2_img1=[];listptimg1img2_img2=[]
    for i1 in range(len(nomvt)):
        for i2 in range(len(nomvt)):
            if i1 < i2:
               transfo1 = np.load(transfo[i1])
               transfo2 = np.load(transfo[i2])
               #print(transfo1,transfo2)
               print('sV1 ',nomvt[i1][-1])
               ptimg1img2_img1, ptimg1img2_img2 = tre_indexes(nomvt[i1],nomvt[i2],transfo1,transfo2,[]) #common points between volumes when no movement
               listptimg1img2_img1.append(ptimg1img2_img1)
               listptimg1img2_img2.append(ptimg1img2_img2)
    
    
    listError=[]
    for i_stack in range(len(images)):
        for i_slice in range(len(images[i_stack])):
            slicei=images[i_stack][i_slice]
            #print(slicei.get_parameters())
            orientation = slicei.get_stackIndex()
            index = slicei.get_indexSlice()
            Errori = sliceFeature(orientation,index)
            listError.append(Errori)
    
    listErrorVolume = separate_features_in_stacks(listError) #list to save error for each slice   
    indice=0;listErrorAfter=[]
    for i1 in range(len(images)):
        for i2 in range(len(images)):
            if i1 < i2:
                transfo1 = np.load(transfo[i1])
                transfo2 = np.load(transfo[i2])
                
                #error of registration between volumes after registration
                errorimg1img2_after = slice_tre(listptimg1img2_img1[indice],listptimg1img2_img2[indice],images[i1],images[i2],listErrorVolume[i1],listErrorVolume[i2])
                indice=indice+1
    
    listErrorAfter=[]
    for n_image in range(len(listErrorVolume)):
         listVolume = listErrorVolume[n_image]
         listv12 = []
         for i_slice in range(len(listVolume)):
             vol = listVolume[i_slice]
             listv12.append(vol.get_median_error())
         listErrorAfter.append(listv12)          

    return listErrorAfter