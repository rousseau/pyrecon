#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 09:59:09 2021

@author: mercier
"""

import numpy as np
from tools import rotationCenter, rigidMatrix


"""
Object that is used for the registration 
The values of this class must not change because they are linked together : use set_parameter to change parameter
A new slice takes into parameters : 
- sliceimage : a 2D  slice extracted from a 3D image, nifti image
- slicemask : a binary image associated with the sliceimage, numpy array  
"""    

class SliceObject: 
    def __init__(self,sliceimage,slicemask,orientation,num_slice,num_image) :
        
        self.__sliceimage = sliceimage  
        self.__parameter = np.zeros(6) #initial parameters of the rigid transform 
        self.__rigid = rigidMatrix(self.__parameter) #initial rigid matrix
        rotC = rotationCenter(slicemask) #Compute the barycenter of the image
        self.__center = np.eye(4) #translation, allow to apply the rigid transform the the center of the image
        self.__invcenter = np.eye(4) #inverse translation 
        center = self.__sliceimage.affine @ rotC
        self.__center[0:3,3] = -center[0:3]
        self.__invcenter[0:3,3] = +center[0:3]
        self.__transfo = self.__invcenter @ (self.__rigid @ (self.__center @ self.__sliceimage.affine)) #translation composed of the affine matrix and the rigid transformation, it gives the new coordinate of the images in word system 
        self.__mask = slicemask #mask associated to the image
        self.__orientation = orientation #image of the slice (0 1 or 2)
        self.__num = num_slice
        self.__image = num_image
        self.ok = 1

    #get and set functions : 
    def get_parameters(self): #return the parameters of the rigid transformation
        return self.__parameter
    
    def get_orientation(self): #return the image of the slice (0 1 or 2)
        return self.__orientation
    
    def set_parameters(self,x): #change the parameters of the rigid transformation
        self.__parameter = x
        self.__rigid = rigidMatrix(x)
        self.__transfo = self.__invcenter @ (self.__rigid @ (self.__center @ self.__sliceimage.affine))
        
    def get_transfo(self): #return the rigid transformation
        return self.__transfo
    
    def get_slice(self): #return the Nifti 2D images
        return self.__sliceimage
    
    def set_slice(self,newSlice): #change the slice
        self.__sliceimage = newSlice
    
    def get_mask(self): #retrun the mask associated to the 2D image
        return self.__mask
    
    def get_center(self):
        return self.__center[0:3,3]
    
    def get_index_slice(self):
        return self.__num
    
    def copy(self):
        copy = SliceObject(self.__sliceimage, self.__mask, self.__orientation,self.__num,self.__image)
        copy.set_parameters(self.__parameter)
        return copy
   
    def get_index_image(self):
        return self.__image
    
