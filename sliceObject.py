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
- sliceimage : a 2D slice extracted from a 3D image
- slicemask : a binary image associated with the sliceimage    
"""    
class SliceObject: 
    def __init__(self,sliceimage,slicemask)    :
        
        self.__sliceimage = sliceimage
        self.__parameter = np.zeros(6)
        self.__rigid = rigidMatrix(self.__parameter)
        rotC = rotationCenter(slicemask,sliceimage.affine) #Compute the barycenter of the image
        self.__center = np.eye(4)
        self.__invcenter = np.eye(4)
        self.__center[0:4,3] = rotC
        self.__invcenter[0:3,3] = -rotC[0:3]
        self.__transfo = self.__invcenter @ (self.__rigid @ (self.__center @ self.__sliceimage.affine))
        self.__mask = slicemask
        

    #get and set functions : 
    def get_parameter(self):
        return self.__parameter
    
    def set_parameter(self,x):
        self.__parameter = x
        self.__rigid = rigidMatrix(self.__parameter)
        print('rigid', self.__rigid)
              
    def get_transfo(self):
        return self.__transfo
    
    def get_slice(self):
        return self.__sliceimage
    
    def get_mask(self):
        return self.__mask