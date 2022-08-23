# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Jun 21 13:42:38 2022

# @author: mercier
# """
import SimpleITK as sitk
import numpy as np
import os
import joblib
from tools import createVolumesFromAlist, rigidMatrix

joblib_name = '/home/mercier/Documents/res/ebner/test/res_sub-0034_ses-0043.joblib.gz'
res = joblib.load(open(joblib_name,'rb'))
list_prefixImage = ['sub-0034_ses-0043_haste_pace_auto_ax_13','sub-0034_ses-0043_haste_pace_auto_cor_12','sub-0034_ses-0043_haste_pace_auto_sag_14','sub-0034_ses-0043_t2_haste_cor_11']

parent_dir = '/home/mercier/Documents/res/ebner/test/'
directory = 'res_sub-0034_ses-0043'
path = os.path.join(parent_dir, directory)
# os.mkdir(path)



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
    rejectedSlice=element[key.index('RejectedSices')] #rejected slices
    
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
                sitk.WriteTransform(test,"%s%s_slice%d.tfm" %(directory,list_prefix,slicei.get_index_slice())) #save rigid transformation, computed at the barycenter of the image, adatpted to itk

convert2EbnerParam(res,list_prefixImage,path)



