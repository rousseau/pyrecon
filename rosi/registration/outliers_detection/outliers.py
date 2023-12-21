#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:31:48 2022

@author: mercier
"""
import numpy as np 
#import registration
import scipy


class sliceFeature: #object that contains all features for outliers classification as well as the slice's error
    
    def __init__(self,stack_num,index_slice):
        self._stack_index = stack_num
        self._index = index_slice
        self._dice = 0
        self._mse = 0
        self._inter = 0
        self._error = 0
        self._nbpoint = 0
        self._slice_error = 0
        self._mask_proportion = 0
        self._bord = False
        self._std_intensity = 0
        self._frequency = 0
        self._center_distance = 0
        self._ncc=-1
        self._mask_point=0

    def add_registration_error(self,new_error):
        self._error=self._error+new_error
        self._nbpoint=self._nbpoint+1
        self._slice_error = self._error/self._nbpoint
     
    def get_error(self):
        return self._slice_error
    
    def reinitialized_error(self):
        self._slice_error=0
        self._error=0
        self._nbpoint=0
    
    def set_error(self,er):
        self._slice_error = er
    
    def set_dice(self,dice):
        self._dice=dice
        
    def get_dice(self):
        return self._dice
    
    def set_mse(self,new_mse):
        self._mse=new_mse
        
    def get_mse(self):
        return self._mse
    
    def set_mask_proportion(self,pmask):
        self._mask_proportion = pmask
        
    def get_mask_proportion(self):
        return self._mask_proportion
        
    def set_bords(self,edge):
        self._bord = edge
        
    def edge(self):
        return self._bord
    
    def set_inter(self,delta):
        self._inter=delta
        
    def get_inter(self):
        return self._inter
    
    def get_stack(self):
        return self._stack_index
    
    def get_index(self):
        return self._index
    
    def get_nbpoint(self):
        return self._nbpoint
    
    def set_nbpoint(self,nb):
        self._nbpoint = nb

    def set_mask_point(self,nb):
        self._mask_point = nb
    
    def get_mask_point(self):
        return self._mask_point
    
    def set_std_intensity(self,std):
        self._std_intensity=std
    
    def get_std_intensity(self):
        return self._std_intensity
    
    def set_frequency(self,freq):
        self._frequency=freq
        
    def get_set_frequency(self):
        return self._frequency
    
    def set_center_distance(self,cd):
        self._center_distance=cd
    
    def get_center_distance(self):
        return self._center_distance
    
    def set_ncc(self,ncc_var):
        self._ncc = ncc_var
    
    def get_ncc(self):
        return self._ncc
        

def separate_features_in_stacks(list_error_feature : 'list[sliceFeature]') -> ('list[sliceFeature]','list[sliceFeature]','list[sliceFeature]'):
   
    """
    From a list of sliceFeature, create several list corresponding to the differents stacks
    """
    
    stack_index = []; list_of_stack=[];

    for s in list_error_feature: 
        
        s_or = s.get_stack()

        if s_or in stack_index: ##check if the list corresponding to the stack has already been created : 
            #if that the case, add the error to the existing list.
            
            index_orientation = stack_index.index(s_or)
            list_of_stack[index_orientation].append(s)
        
        else: #if not, a new list is created
            stack_index.append(s_or)
            list_of_stack.append([])
            index_orientation = stack_index.index(s_or)
            list_of_stack[index_orientation].append(s)
                
    return list_of_stack